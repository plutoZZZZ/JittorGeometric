/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-07-03 13:50:09
 */

#include  "var.h"
#include  <cuda.h>
#include  "edgesoftmax_op.h"
#ifdef JIT_cuda
// #include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/cub.cuh>
#endif

typedef uint32_t VertexId_CUDA;
const int CUDA_NUM_THREADS_SOFTMAX = 32;
const int CUDA_NUM_BLOCKS_SOFTMAX = 512;

namespace jittor {
#ifndef JIT
EdgesoftmaxOp::EdgesoftmaxOp(Var* outputVar_, Var* x_, Var* indices_,Var* offset_) :
outputVar(outputVar_),x(x_),indices(indices_),offset(offset_){
    flags.set(NodeFlags::_cpu, 1);
    flags.set(NodeFlags::_cuda, 1);
    output = create_output(nullptr,x->dtype());
}


void EdgesoftmaxOp::jit_prepare(JK& jk) {
     add_jit_define(jk, "T", x->dtype());
     add_jit_define(jk, "Tint", indices->dtype());
}

#else // JIT
    #ifdef JIT_cpu
    void EdgesoftmaxOp::jit_run() {
        auto* __restrict__ out_ptr = outputVar->ptr<T>();
        auto* __restrict__ x_ptr = x->ptr<T>();
        auto* __restrict__ i_ptr = indices->ptr<Tint>();
        auto* __restrict__ o_ptr = offset->ptr<Tint>();
        int e_num=indices->shape[0];
        int v_num=offset->shape[0]-1;
        int feature_dim=x->shape[1];
        int start;
        int end;
        
        // For each vertex (column in CSC)
        for(int vtx=0;vtx<v_num;vtx++){
            start=o_ptr[vtx];
            end=o_ptr[vtx+1];
            
            // For each feature dimension (independent softmax)
            for(int f=0;f<feature_dim;f++){
                T max_weight = -1e20;
                T total = 0;
                
                // Find max for numerical stability
                for(int i=start;i<end;i++){
                    T val = x_ptr[i * feature_dim + f];
                    if(val > max_weight) {
                        max_weight = val;
                    }
                }
                
                // Compute exp(x - max) and sum
                for(int i=start;i<end;i++){
                    T val = x_ptr[i * feature_dim + f];
                    out_ptr[i * feature_dim + f] = exp(val - max_weight);
                    total += out_ptr[i * feature_dim + f];
                }
                
                // Normalize
                for(int i=start;i<end;i++){
                    out_ptr[i * feature_dim + f] /= total;
                }
            }
        }
    }
    #else //cuda
        template <typename T_v,typename T_l>
        __global__ void edge_softmax_forward_block( T_v* msg_output,  T_v* msg_input,
                         const T_l *row_indices, const T_l *column_offset,
                         T_l batch_size_, T_l feature_size_) {
            int VtxPerBlock = 1;
            typedef ::cub::BlockReduce<T_v, CUDA_NUM_THREADS_SOFTMAX> BlockReduce;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            __shared__ T_v bcast_sum;
            __shared__ T_v bcast_max;
            
            for (VertexId_CUDA blkColStart = blockIdx.x * VtxPerBlock; 
                 blkColStart < batch_size_; 
                 blkColStart += VtxPerBlock * gridDim.x) {
                
                VertexId_CUDA curVtx_trans = blkColStart;
                VertexId_CUDA rowIdxStart = column_offset[curVtx_trans];
                VertexId_CUDA rowIdxEnd = column_offset[curVtx_trans + 1];
                
                // Process each feature dimension independently
                for (T_l f = 0; f < feature_size_; f++) {
                    __syncthreads();
                    
                    // Step 1: Find max value for numerical stability
                    T_v thread_max = -1e20;
                    for (VertexId_CUDA eid = rowIdxStart + threadIdx.x; 
                         eid < rowIdxEnd; 
                         eid += CUDA_NUM_THREADS_SOFTMAX) {
                        T_v val = msg_input[eid * feature_size_ + f];
                        if (val > thread_max) {
                            thread_max = val;
                        }
                    }
                    
                    T_v block_max = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());
                    
                    if (threadIdx.x == 0) {
                        bcast_max = block_max;
                    }
                    __syncthreads();
                    
                    // Step 2: Compute exp(x - max) and sum
                    T_v thread_sum = 0.0;
                    for (VertexId_CUDA eid = rowIdxStart + threadIdx.x; 
                         eid < rowIdxEnd; 
                         eid += CUDA_NUM_THREADS_SOFTMAX) {
                        T_v val = msg_input[eid * feature_size_ + f];
                        T_v exp_val = exp(val - bcast_max);
                        msg_output[eid * feature_size_ + f] = exp_val;
                        thread_sum += exp_val;
                    }
                    
                    T_v block_sum = BlockReduce(temp_storage).Sum(thread_sum);
                    
                    if (threadIdx.x == 0) {
                        bcast_sum = block_sum;
                    }
                    __syncthreads();
                    
                    // Step 3: Normalize
                    for (VertexId_CUDA eid = rowIdxStart + threadIdx.x; 
                         eid < rowIdxEnd; 
                         eid += CUDA_NUM_THREADS_SOFTMAX) {
                        msg_output[eid * feature_size_ + f] /= bcast_sum;
                    }
                }
            }
        }
        
        void EdgesoftmaxOp::jit_run() {
            auto* __restrict__ out_ptr = outputVar->ptr<T>();
            auto* __restrict__ x_ptr = x->ptr<T>();
            auto* __restrict__ i_ptr = indices->ptr<Tint>();
            auto* __restrict__ o_ptr = offset->ptr<Tint>();
            Tint e_num = indices->shape[0];
            Tint v_num = offset->shape[0] - 1;
            Tint size = x->shape[1];
            
            edge_softmax_forward_block<T, Tint><<<CUDA_NUM_BLOCKS_SOFTMAX, CUDA_NUM_THREADS_SOFTMAX>>>(
                out_ptr, x_ptr, i_ptr, o_ptr, v_num, size);
        }
    #endif //cuda
#endif // JIT

} // jittor