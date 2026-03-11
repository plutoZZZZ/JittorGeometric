'''
Description: 
Author: lusz
Date: 2024-07-03 13:50:35
'''
import jittor as jt
import os
import sys
from jittor import nn
from jittor import Function
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from jittor_geometric.data import CSC, CSR


# class EdgeSoftMaxFunc(Function):
#     def execute(self,x,csc):
#         v_num = jt.size(csc.column_offset, 0) - 1
#         out = None
#         for vtx in range(v_num):
#             start = csc.column_offset[vtx]
#             end = csc.column_offset[vtx + 1]
#             slice = x[start:end]
#             softmax_slice = jt.nn.softmax(slice) 
#             if out is None:
#                 out = softmax_slice
#             else:
#                 out = jt.contrib.concat([out, softmax_slice], dim=0)
#         self.y = out
#         self.csc = csc
#         return out

#     def grad(self, grad_output):
#         # print(grad_output)
#         v_num = jt.size(self.csc.column_offset, 0) - 1
#         output_grad = None 
#         for vtx in range(v_num):
#             start = self.csc.column_offset[vtx]
#             end = self.csc.column_offset[vtx + 1]
#             slice = grad_output[start:end] 
#             imr = self.y[start:end]
#             d_o = imr * slice - imr * (slice.sum() * imr)
#             if output_grad is None:
#                 output_grad = d_o
#             else:
#                 output_grad = jt.contrib.concat([output_grad, d_o], dim=0)
#         return output_grad, None
    
# def EdgeSoftmax(x,csc):
#     out = EdgeSoftMaxFunc.apply(x,csc)
#     return out

module_path = os.path.dirname(__file__)
src = os.path.join(module_path, "cpp/edgesoftmax_op.cc")
header = os.path.join(module_path, "cpp/edgesoftmax_op.h")
edge_softmax_op = jt.compile_custom_ops((src, header))

src_b = os.path.join(module_path, "cpp/edgesoftmaxbackward_op.cc")
header_b = os.path.join(module_path, "cpp/edgesoftmaxbackward_op.h")
edge_softmax_backward_op = jt.compile_custom_ops((src_b, header_b))


class EdgeSoftmaxFunc(Function):
    def execute(self,x,csc):
        self.x=x
        self.csc=csc
        self.was_1d = (len(x.shape) == 1)
        
        # Ensure input is always 2D for C++ implementation
        if self.was_1d:
            x = x.reshape(-1, 1)
            
        output=jt.zeros_like(x)
        edge_softmax_op.edgesoftmax(output,x,csc.row_indices,csc.column_offset)
        self.y=output
        
        # Restore original shape if needed
        if self.was_1d:
            output = output.reshape(-1)
            
        return output

    def grad(self, grad_output):
        y = self.y
        if self.was_1d:
            grad_output = grad_output.reshape(-1, 1)
            
        output_grad=jt.zeros_like(grad_output)
        edge_softmax_backward_op.edgesoftmaxbackward(output_grad,grad_output,y,self.csc.row_indices,self.csc.column_offset)
        
        if self.was_1d:
            output_grad = output_grad.reshape(-1)
            
        return output_grad,None
    

def EdgeSoftmax(x,csc):
    out = EdgeSoftmaxFunc.apply(x,csc)
    return out