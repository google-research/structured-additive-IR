#ifndef THIRD_PARTY_SAIR_LOOP_NEST_H_
#define THIRD_PARTY_SAIR_LOOP_NEST_H_

#include "mlir/Support/LogicalResult.h"
#include "sair_op_interfaces.h"
#include "sair_ops.h"

namespace sair {

// Verifies loop nest attributes of operations nested in the
// sair.program operation.
mlir::LogicalResult VerifyLoopNests(SairProgramOp program);

// Verifies the loop nest attribute of the operation.
mlir::LogicalResult VerifyLoopNest(ComputeOp op);

}  // namespace sair

#endif  // THIRD_PARTY_SAIR_LOOP_NEST_H_
