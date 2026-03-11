/*
 * @Description: 
 * @Author: lusz
 * @Date: 2024-07-04 16:16:13
 */

#include "var.h"
#include "edgesoftmaxbackward_op.h"
typedef uint32_t VertexId_CUDA;
const int CUDA_NUM_THREADS_SOFTMAX = 32;
const int CUDA_NUM_BLOCKS_SOFTMAX = 512;

namespace jittor {
#ifndef JIT
EdgesoftmaxbackwardOp::EdgesoftmaxbackwardOp(Var* outputVar_, Var* x_, Var* y_,Var* indices_,Var* offset_) :
outputVar(outputVar_),x(x_),y(y_),indices(indices_),offset(offset_){
    flags.set(NodeFlags::_cpu, 1);
    output = create_output(nullptr,x->dtype());
}


void EdgesoftmaxbackwardOp::jit_prepare(JK& jk) {
     add_jit_define(jk, "T", x->dtype());
     add_jit_define(jk, "Tint", indices->dtype());
}

#else // JIT
    #ifdef JIT_cpu
    void EdgesoftmaxbackwardOp::jit_run() {
        auto* __restrict__ out_ptr = outputVar->ptr<T>();
        auto* __restrict__ x_ptr = x->ptr<T>();  // grad_output
        auto* __restrict__ y_ptr = y->ptr<T>();  // y (softmax output)
        auto* __restrict__ i_ptr = indices->ptr<int>();
        auto* __restrict__ o_ptr = offset->ptr<int>();
        int e_num = indices->shape[0];
        int v_num = offset->shape[0] - 1;
        int feature_dim = x->shape[1];
        int start;
        int end;
        
        for (int vtx = 0; vtx < v_num; vtx++) {
            start = o_ptr[vtx];
            end = o_ptr[vtx + 1];
            
            // For each feature dimension
            for (int f = 0; f < feature_dim; f++) {
                T dot = 0.0f;
                
                // Compute dot = sum(y_i * grad_output_i)
                for (int i = start; i < end; ++i) {
                    dot += y_ptr[i * feature_dim + f] * x_ptr[i * feature_dim + f];
                }
                
                // Compute gradient: grad_input = y_i * (grad_output_i - dot)
                for (int i = start; i < end; ++i) {
                    out_ptr[i * feature_dim + f] = y_ptr[i * feature_dim + f] * 
                        (x_ptr[i * feature_dim + f] - dot);
                }
            }
        }
    }
    #else //cuda
    template <typename T_v, typename T_l>
    __global__ void edge_softmax_backward_block(T_v* msg_input_grad, T_v* msg_output_grad,
                    T_v* msg_cached, const T_l *row_indices, const T_l *column_offset,
                    T_l batch_size_, T_l feature_size_) {
        int VtxPerBlock = 1;
        typedef cub::BlockReduce<T_v, CUDA_NUM_THREADS_SOFTMAX> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        __shared__ T_v bcast_dot;
        
        for (VertexId_CUDA blkColStart = blockIdx.x * VtxPerBlock; 
             blkColStart < batch_size_; 
             blkColStart += VtxPerBlock * gridDim.x) {
            
            VertexId_CUDA curVtx_trans = blkColStart;
            VertexId_CUDA rowIdxStart = column_offset[curVtx_trans];
            VertexId_CUDA rowIdxEnd = column_offset[curVtx_trans + 1];
            
            // Process each feature dimension independently
            for (T_l f = 0; f < feature_size_; f++) {
                __syncthreads();
                
                // Compute dot = sum(grad_output * y) for this vertex and feature
                T_v thread_dot = 0.0;
                for (VertexId_CUDA eid = rowIdxStart + threadIdx.x; 
                     eid < rowIdxEnd; 
                     eid += CUDA_NUM_THREADS_SOFTMAX) {
                    thread_dot += msg_output_grad[eid * feature_size_ + f] * 
                                  msg_cached[eid * feature_size_ + f];
                }
                
                T_v block_dot = BlockReduce(temp_storage).Sum(thread_dot);
                
                if (threadIdx.x == 0) {
                    bcast_dot = block_dot;
                }
                __syncthreads();
                
                // Compute gradient: grad_input = y * (grad_output - dot)
                for (VertexId_CUDA eid = rowIdxStart + threadIdx.x; 
                     eid < rowIdxEnd; 
                     eid += CUDA_NUM_THREADS_SOFTMAX) {
                    T_v y_val = msg_cached[eid * feature_size_ + f];
                    T_v grad_val = msg_output_grad[eid * feature_size_ + f];
                    msg_input_grad[eid * feature_size_ + f] = y_val * (grad_val - bcast_dot);
                }
            }
        }
    }

    void EdgesoftmaxbackwardOp::jit_run() {
        auto* __restrict__ out_ptr = outputVar->ptr<T>();
        auto* __restrict__ x_ptr = x->ptr<T>();  // grad_output
        auto* __restrict__ y_ptr = y->ptr<T>();  // y (softmax output)
        auto* __restrict__ i_ptr = indices->ptr<int>();
        auto* __restrict__ o_ptr = offset->ptr<int>();
        Tint e_num = indices->shape[0];
        Tint v_num = offset->shape[0] - 1;
        Tint size = x->shape[1];
        
        edge_softmax_backward_block<T, Tint><<<CUDA_NUM_BLOCKS_SOFTMAX, CUDA_NUM_THREADS_SOFTMAX>>>(
            out_ptr, x_ptr, y_ptr, i_ptr, o_ptr, v_num, size);
    }
    #endif// cuda
#endif // JIT

} // jittor