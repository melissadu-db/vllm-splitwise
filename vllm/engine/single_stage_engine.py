from __future__ import annotations
import time, copy
from typing import Callable, Optional, List, Dict, Tuple
from abc import ABC, abstractmethod
import asyncio

from vllm.core.scheduler import Scheduler, SchedulerType
from vllm.utils import Stage
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig, LoRAConfig)
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import PlacementGroup
from vllm.sequence import (Logprob, SamplerOutput, Sequence, SequenceGroup,
                           SequenceGroupOutput, SequenceOutput, SequenceStatus)
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple,
                    Type, Union, AsyncGenerator)

class SingleStageLLMEngine(ABC):
    """
    SingleStageLLMEngine: An LLMEngine that runs either the context stage or the decoding stage.
    
    This class is the base class for ContextStageLLMEngine and DecodingStageLLMEngine.
    """
    def _free_request_resources(self, request_id: int) -> None:
        self.block_manager.free_blocks(request_id)
        self._remote_call_all_workers_async("clear_request_resource", request_id)
    
    def __init__(
        self,
        stage: Stage,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        sched_config: SchedulerConfig,
        placement_groups: List[PlacementGroup],
        model_executor,
        # engine_on_new_step_output_callback: Callable[[int, StepOutput], None],   # The LLMEngine's callback function when a new StepOutput of a particular request is generated
        # engine_on_new_lifetime_event_callback: Optional[Callable[[int, LifetimeEvent, bool], None]] = None,   # The LLMEngine's callback function when a new LifetimeEvent of a particular request is generated
    ):
        self.stage = stage
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.cache_config = cache_config
        self.sched_config = sched_config
        # self.engine_on_new_step_output_callback = engine_on_new_step_output_callback
        # self.engine_on_new_lifetime_event_callback = engine_on_new_lifetime_event_callback

        self.placement_groups = placement_groups
        self.model_executor = model_executor
        
        # workers[i][j] is the j-th tensor-parallel worker in pipeline stage i
        self.workers = []
        
class PrefillLLMEngine(SingleStageLLMEngine):
    def __init__(
        self,
        bridge_queue: asyncio.Queue[SequenceGroup],
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        sched_config: SchedulerConfig,
        placement_groups: List[PlacementGroup],
        model_executor,
        # engine_on_new_step_output_callback: Callable[[int, RequestOutput], None],
        # engine_on_new_lifetime_event_callback: Callable[[int, LifetimeEvent, bool], None]
    ):

        super().__init__(
            Stage.CONTEXT,
            model_config,
            parallel_config,
            cache_config,
            sched_config,
            placement_groups,
            model_executor,
            # engine_on_new_step_output_callback,
            # engine_on_new_lifetime_event_callback
        )

        self.scheduler = Scheduler(
            SchedulerType.PREFILL,
            sched_config,
            cache_config,
        )
        
        # All the batchedrequests that are pushed into the pipeline
        # Note: len(batched_in_pipeline) <= pp_size and batches are appended in FIFO
        self.batches_in_pipeline: List[BatchedRequests] = []
        self.batches_ret_futures = []  
        self.bridge_queue = bridge_queue
    
    def add_request(self, request: Request):
        self.scheduler.add_request(request)
    
    def _free_request_resources(self, request_id: int):
        super()._free_request_resources(request_id)
        
    async def _step(self):
        """
        Run one step of inference on the batch of requests chosen by the scheduler.
        
        Note: if pipeline parallelism is used, one step only kicks one stage of execution,
        and each request needs #pp steps in total to generate one token.
        
        Note2. Pipeline parallel is not tested yet
        """
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()

        if scheduler_outputs.is_empty():
            output = []
        else:
            output = self.model_executor.execute_model(
                seq_group_metadata_list, scheduler_outputs.blocks_to_swap_in,
                scheduler_outputs.blocks_to_swap_out,
                scheduler_outputs.blocks_to_copy,
                scheduler_outputs.blocks_to_nw)
        
        if outputs:
            self.bridge_queue.put_nowait(migrating_req) # This won't panic because the queue is unbounded

        # if len(self.batches_in_pipeline) == self.parallel_config.pipeline_parallel_size:
        #     # if the pipeline is full, block until the earliest batch returns
        #     # if pipeline parallelism is not used, i.e., pp = 1, this should always be true
        #     if self.batches_ret_futures[0] is None:
        #         # No request in the batch
        #         self.batches_in_pipeline.pop(0)
        #         self.batches_ret_futures.pop(0)
        #     else:
        #         generated_tokens_ids = await self.batches_ret_futures[0]
                    
        #         end_time = time.time()
        #         generated_tokens = []
        #         for gen_token_id in generated_tokens_ids:
        #             try:
        #                 token = self.tokenizer.decode(gen_token_id)
        #             except Exception as e:
        #                 print(f"(context) Warning: Cannot decode token with id {gen_token_id}. Error: {e}")
        #                 token = ""
        #             generated_tokens.append(token)

        #         finished_batch = self.batches_in_pipeline[0]
        #         finished_batch.finish_one_iteration(
        #             generated_tokens, generated_tokens_ids, end_time
        #         )
                
        #         self.scheduler.on_finish_requests(finished_batch)
                
        #         for request, new_token, new_token_id in zip(
        #             finished_batch.requests, generated_tokens, generated_tokens_ids
        #         ):
        #             step_output = RequestOutput(request, new_token, new_token_id)
        #             self.engine_on_new_lifetime_event_callback(
        #                 request.request_id,
        #                 LifetimeEvent(LifetimeEventType.ContextEnd)
        #             )
        #             self.engine_on_new_step_output_callback(
        #                 request.request_id,
        #                 step_output
        #             )

        #         # Cannot free blocks now! The decoding stage may still need them!

        #         self.batches_in_pipeline.pop(0)
        #         self.batches_ret_futures.pop(0)
                
        #         # Inform the user that the request has finished the context stage
        #         for request in finished_batch.requests:
        #             if not request.is_finished:
        #                 # Push the request into the bridge queue if it is not finished
        #                 migrating_req = MigratingRequest(
        #                     request,
        #                     self.block_manager.get_block_table(request.request_id),
        #                     self.parallel_config,
        #                 )
        #                 self.bridge_queue.put_nowait(migrating_req) # This won't panic because the queue is unbounded
        #             else:
        #                 self._free_request_resources(request.request_id)

        return self._process_model_outputs(output, scheduler_outputs)
    
    # def clear_migrated_blocks_callback(self, migrated_request: MigratingRequest):
    #     """
    #     Called when the decoding engine finishes migrating the blocks of the request.
    #     """
    #     self._free_request_resources(migrated_request.req.request_id)
    #     self.scheduler.on_request_migrated(migrated_request)
        
    async def start_event_loop(self):
        async def event_loop1():
            while True:
                await self._step()
                await asyncio.sleep(SLEEP_IN_EACH_EVENT_LOOP)
        
        async def event_loop2():
            while True:
                self.print_engine_status()
                await asyncio.sleep(PRINT_STATUS_INTERVAL)

        await asyncio.gather(event_loop1(), event_loop2())
        
    def print_engine_status(self):
        self.scheduler.print_status()

class DecodeLLMEngine(SingleStageLLMEngine):
    def __init__(
        self,
        bridge_queue: asyncio.Queue[SequenceGroup],
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        cache_config: CacheConfig,
        sched_config: SchedulerConfig,
        placement_groups: List[PlacementGroup],
        model_executor,
        # clear_migrated_blocks_callback: Callable[[Request], None],
        # engine_on_new_step_output_callback: Callable[[int, StepOutput], None],
        # engine_on_new_lifetime_event_callback: Callable[[int, LifetimeEvent, bool], None]
    ):

        super().__init__(
            Stage.DECODING,
            model_config,
            parallel_config,
            cache_config,
            sched_config,
            placement_groups,
            model_executor,
            # engine_on_new_step_output_callback,
            # engine_on_new_lifetime_event_callback
        )

        self.scheduler = Scheduler(
            SchedulerType.DECODE,
            sched_config,
            cache_config,
        )
        
        self.bridge_queue = bridge_queue
        # self.clear_migrated_blocks_callback = clear_migrated_blocks_callback
        
        # All the batchedrequests that are pushed into the pipeline
        # Note: len(batched_in_pipeline) <= pp_size and batches are appended in FIFO
        self.batches_in_pipeline = []
        self.batches_ret_futures = []
    
    def _free_request_resources(self, request_id: int):
        super()._free_request_resources(request_id)
        self.request_events.pop(request_id)
        self.request_outputs.pop(request_id)
            
    async def _step(self) -> None:
        """
        Run one step of inference on the batch of requests chosen by the scheduler.
        Note: if pipeline parallelism is used, one step only kicks one stage of execution,
        and each request needs #pp steps in total to generate one token.
        """

        pp_size = self.parallel_config.pipeline_parallel_size
        tp_size = self.parallel_config.tensor_parallel_size

        # pick next batch from scheduler
        # this may trigger migration if some requests are still at context stage
        # this may trigger swap_in if some requests have been swapped out to CPU
        # this may also trigger swap_out if GPU blocks are not enough
        # batched_requests = self.scheduler.get_next_batch()

        # if len(batched_requests) == 0:
        #     self.batches_in_pipeline.append(batched_requests)
        #     self.batches_ret_futures.append(None)
        #     await asyncio.sleep(SLEEP_WHEN_DECODING_NO_REQUEST)
        # else:
        #     # Log down the lifetime event
        #     for request in batched_requests.requests:
        #         self.engine_on_new_lifetime_event_callback(
        #             request.request_id,
        #             LifetimeEvent(LifetimeEventType.DecodingBegin),
        #             True
        #         )
                
        #     # Allocate blocks as needed
        #     self.block_manager.allocate_blocks_batched(batched_requests)

        #     # Check if all requests are on GPU (i.e. not swapped out)
        #     assert self.block_manager.is_all_requests_on_gpu(
        #         batched_requests
        #     ), "Some requests are currently swapped out to CPU"

        #     # push the batch into pipeline
        #     batched_requests.start_one_iteration(time.time())
        #     self.batches_in_pipeline.append(batched_requests)
        #     remote_calls = self._remote_call_all_workers_async(
        #         "step",
        #         batched_requests.get_request_ids(),
        #         batched_requests.get_input_tokens_batched(),
        #         batched_requests.get_first_token_indexes(),
        #         self.block_manager.get_partial_block_table(
        #             batched_requests.get_request_ids()
        #         ),
        #     )
        #     # only the leader of the last stage return valid output, i.e., generated tokens ids
        #     self.batches_ret_futures.append(remote_calls[(pp_size - 1) * tp_size])

        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()

        if scheduler_outputs.is_empty():
            output = []
        else:
            output = self.model_executor.execute_model(
                seq_group_metadata_list, scheduler_outputs.blocks_to_swap_in,
                scheduler_outputs.blocks_to_swap_out,
                scheduler_outputs.blocks_to_copy,
                scheduler_outputs.blocks_to_nw)

        # output buffer
        finished_reqs = []

        # if len(self.batches_in_pipeline) == self.parallel_config.pipeline_parallel_size:
        #     # if the pipeline is full, block until the earliest batch returns
        #     # if pipeline parallelism is not used, i.e., pp = 1, this should always be true
        #     if self.batches_ret_futures[0] is None:
        #         self.batches_in_pipeline.pop(0)
        #         self.batches_ret_futures.pop(0)
        #     else:
        #         generated_tokens_ids = await self.batches_ret_futures[0]
        #         end_time = time.time()
        #         generated_tokens = []
        #         for gen_token_id in generated_tokens_ids:
        #             try:
        #                 token = self.tokenizer.decode(gen_token_id)
        #             except Exception as e:
        #                 print(f"(decoding) Warning: Cannot decode token with id {gen_token_id}. Error: {e}")
        #                 token = ""
        #             generated_tokens.append(token)

        #         finished_batch = self.batches_in_pipeline[0]
        #         finished_batch.finish_one_iteration(
        #             generated_tokens, generated_tokens_ids, end_time
        #         )

        #         for request, new_token, new_token_id in zip(
        #             finished_batch.requests, generated_tokens, generated_tokens_ids
        #         ):
        #             self.engine_on_new_step_output_callback(
        #                 request.request_id,
        #                 StepOutput(request, new_token, new_token_id)
        #             )
        #             if request.is_finished:
        #                 self.engine_on_new_lifetime_event_callback(
        #                     request.request_id,
        #                     LifetimeEvent(LifetimeEventType.DecodingEnd)
        #                 )
        #         finished_reqs = self.scheduler.pop_finished_requests()

        #         # free blocks for finished requests
        #         self.block_manager.free_blocks_batched(finished_reqs)
        #         self._remote_call_all_workers_async(
        #             "clear_request_resource_batched", finished_reqs
        #         )

        #         # pop the finished batch
        #         self.batches_in_pipeline.pop(0)
        #         self.batches_ret_futures.pop(0)

        # # proactive request migraion
        # await self.scheduler.post_process()
        return self._process_model_outputs(output, scheduler_outputs)
    
    async def start_event_loop(self):
        async def event_loop1():
            # Event loop 1. Add migrating request to the scheduler
            while True:
                migrating_req = await self.bridge_queue.get()
                await self.scheduler.add_request(migrating_req)
                self.bridge_queue.task_done()
        
        async def event_loop2():
            # Event loop 2. Run step()
            while True:
                await self._step()
                await asyncio.sleep(SLEEP_IN_EACH_EVENT_LOOP)
        
        async def event_loop3():
            # Event loop 3. Print engine status
            while True:
                self.print_engine_status()
                await asyncio.sleep(PRINT_STATUS_INTERVAL)
                
        await asyncio.gather(event_loop1(), event_loop2(), event_loop3())
    
    def print_engine_status(self):
        self.block_manager.print_block_usage()
        self.scheduler.print_status()
        