import torch
import cupy
from cupy_backends.cuda.libs import nccl
import torch.distributed as dist
from typing import List


def _type_torch_to_cupy(torch_type: torch.dtype):
    # print(torch_type)
    mappings = {
        torch.uint8: nccl.NCCL_UINT8,
        torch.int32: nccl.NCCL_INT32,
        torch.int: nccl.NCCL_INT,
        torch.float16: nccl.NCCL_FLOAT16,
        torch.float32: nccl.NCCL_FLOAT32,
        torch.float64: nccl.NCCL_FLOAT64,
        torch.float: nccl.NCCL_FLOAT
    }
    return mappings[torch_type]


class NCCLCommunicator:
    def __init__(self,
                 comm_rank: int,
                 cuda_id: int,
                 comm_group_size: int,
                 comm_name: str):
        self.comm_rank = comm_rank
        cupy.cuda.Device(cuda_id).use()
        self.comm_group_size = comm_group_size
        print("Initialize NCCLCommunicator: <", comm_name, ">; rank:", comm_rank)
        self.dist_store = dist.distributed_c10d._get_default_store()

        if self.comm_rank == 0:
            # cupy >= 13: get_unique_id() returns bytes, which NcclCommunicator also expects.
            cuda_id_str = nccl.get_unique_id()
            self.dist_store.set('group-'+comm_name+'-unique-id', cuda_id_str)
        else:
            cuda_id_str = self.dist_store.get('group-'+comm_name+'-unique-id')

        self.comm = nccl.NcclCommunicator(comm_group_size, bytes(cuda_id_str), comm_rank)

    @staticmethod
    def barrier():
        dist.barrier()

    def store_set(self, key, value):
        self.dist_store.set(key, value)

    def store_get(self, key):
        return self.dist_store.get(key)

    def send(self,
             tensor: torch.Tensor,
             dst: int,
             stream=cupy.cuda.Stream.null):
        # print("Send tensor of size:", torch.numel(tensor))
        self.comm.send(
            tensor.data_ptr(),
            torch.numel(tensor),
            _type_torch_to_cupy(tensor.dtype),
            dst,
            stream.ptr
        )

    def recv(self,
             tensor: torch.Tensor,
             src: int,
             stream=cupy.cuda.Stream.null):
        # print("Recv tensor of size:", torch.numel(tensor))
        # print("mean:", torch.mean(tensor).item(), " std:", torch.std(tensor).item())
        self.comm.recv(
            tensor.data_ptr(),
            torch.numel(tensor),
            _type_torch_to_cupy(tensor.dtype),
            src,
            stream.ptr
        )

    def broadcast(self,
                  tensor: torch.Tensor,
                  src: int,
                  stream=cupy.cuda.Stream.null):
        self.comm.bcast(
            tensor.data_ptr(),
            torch.numel(tensor),
            _type_torch_to_cupy(tensor.dtype),
            src,
            stream.ptr
        )

    def reduce(self,
               tensor: torch.Tensor,
               dst: int,
               stream=cupy.cuda.Stream.null,
               op=nccl.NCCL_SUM):
        self.comm.reduce(
            tensor.data_ptr(),  # force it to be in-place.
            tensor.data_ptr(),
            torch.numel(tensor),
            _type_torch_to_cupy(tensor.dtype),
            op,
            dst,
            stream.ptr
        )

    def all_reduce(self,
                  tensor: torch.Tensor,
                  stream=cupy.cuda.Stream.null,
                  op=nccl.NCCL_SUM):
        self.comm.allReduce(
            tensor.data_ptr(),
            tensor.data_ptr(),
            torch.numel(tensor),
            _type_torch_to_cupy(tensor.dtype),
            op,
            stream.ptr
        )

    def scatter(self,
                tensor: torch.Tensor,
                scatter_list: List[torch.Tensor],
                src: int,
                stream=cupy.cuda.Stream.null):
        nccl.groupStart()
        if self.comm_rank == src:
            for i in range(self.comm_group_size):
                self.send(
                    scatter_list[i],
                    i,
                    stream
                )
        self.recv(
            tensor,
            src,
            stream
        )
        nccl.groupEnd()

    def gather(self,
               tensor: torch.Tensor,
               gather_list: List[torch.Tensor],
               dst: int,
               stream=cupy.cuda.Stream.null):
        nccl.groupStart()
        if self.comm_rank == dst:
            for i in range(self.comm_group_size):
                self.recv(
                    gather_list[i],
                    i,
                    stream
                )
        self.send(
            tensor,
            dst,
            stream
        )
        nccl.groupEnd()

    def all_to_all(self,
                   output_tensor_list: List[torch.Tensor],
                   input_tensor_list: List[torch.Tensor],
                   stream=cupy.cuda.Stream.null):
        assert len(output_tensor_list) == self.comm_group_size and len(input_tensor_list) == self.comm_group_size
        nccl.groupStart()
        for i in range(self.comm_group_size):
            self.send(input_tensor_list[i], i, stream)
            self.recv(output_tensor_list[i], i, stream)
        nccl.groupEnd()

    def all_gather(self,
                   tensor: torch.Tensor,
                   output_tensor_list: List[torch.Tensor],
                   stream=cupy.cuda.Stream.null
                   ):
        assert len(output_tensor_list) == self.comm_group_size
        nccl.groupStart()
        for i in range(self.comm_group_size):
            self.send(tensor, i, stream)
            self.recv(output_tensor_list[i], i, stream)
        nccl.groupEnd()

    def all_reduce_opt(self,
                       tensor: torch.Tensor,
                       buffer: List[torch.Tensor],
                       stream=cupy.cuda.Stream.null):
        # First do all-to-all
        assert torch.numel(tensor.data) % self.comm_group_size == 0
        chunk_size = torch.numel(tensor.data) // self.comm_group_size
        t_type = _type_torch_to_cupy(tensor.dtype)
        element_size = tensor.data.element_size()
        nccl.groupStart()
        for i in range(self.comm_group_size):
            self.comm.send(tensor.data_ptr()+i*chunk_size*element_size, chunk_size, t_type, i, stream.ptr)
            self.comm.recv(buffer[i].data_ptr(), chunk_size, t_type, i, stream.ptr)
        nccl.groupEnd()
        for i in range(1, self.comm_group_size):
            buffer[0] += buffer[i]
        nccl.groupStart()
        for i in range(self.comm_group_size):
            self.comm.send(buffer[0].data_ptr(), chunk_size, t_type, i, stream.ptr)
            self.comm.recv(tensor.data_ptr()+i*chunk_size*element_size, chunk_size, t_type, i, stream.ptr)
        nccl.groupEnd()


def default_init(args):
    dist.init_process_group(backend='gloo', init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

