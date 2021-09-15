// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SAIR_MAPPED_DOMAIN_H_
#define SAIR_MAPPED_DOMAIN_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "sair_attributes.h"
#include "sair_op_interfaces.h"

namespace sair {

class LoopNest;

// A domain mapped into a new domain by a mapping attribute.
//
// Both source and target domains are defined in a loop nest that is also
// maintained by this class.
class MappedDomain : public AttrLocation {
 public:
  // Create a 0-dimension mapped domain defined in `loop_nest`.
  MappedDomain(mlir::Location loc, llvm::StringRef kind, mlir::StringAttr name,
               const LoopNest &loop_nest);

  // Source domain. This is prefixed by the loop nest domain.
  llvm::ArrayRef<ValueAccessInstance> domain() const { return domain_; }

  // Mapping from the source domain to the new domain.
  MappingAttr mapping() const { return mapping_; }

  // Name of the loops the mapped domain is defined in.
  llvm::ArrayRef<mlir::StringAttr> loop_nest() const { return loop_nest_; }

  // Shape of the source domain.
  DomainShapeAttr DomainShape() const;

  // Mapping prefixed with the mapping from domain to loops.
  MappingAttr NestedMapping() const;

  // Shape of the target domain prefixed with its loop nest shape.
  DomainShapeAttr NestedShape() const;

  // Extends the mapping by adding none expressions at the front.
  void AddNonePrefixToMapping(int new_dimensions);

  // Unifies the mapping with another mapping, extending the domain as needed.
  // Emits an error if mappings cannot be unified. `loop_nest_mapping` is a
  // mapping from the new mapping domain to the loop nest domain.
  mlir::LogicalResult UnifyMapping(
      const OpInstance &op, MappingAttr loop_nest_mapping,
      MappingAttr new_mapping,
      llvm::ArrayRef<ValueAccessInstance> new_mapping_domain);

  // Updates the loop nest in which the target domain is defined. The new loop
  // nest must be a prefix of the current one.
  void SetLoopNest(const LoopNest &new_loop_nest);

 private:
  // Extends the domain to containt `dimension` if it does not already contain
  // it. Updates `constraint` with the new expression pointing to `dimension`.
  // Emits an error on failure.
  mlir::LogicalResult ResolveUnification(const OpInstance &op, int dimension_id,
                                         const ValueAccessInstance &dimension,
                                         MappingExpr &constraint);

  llvm::SmallVector<ValueAccessInstance> domain_;
  MappingAttr mapping_;

  llvm::SmallVector<mlir::StringAttr> loop_nest_;
  MappingAttr loops_mapping_;
};

}  // namespace sair

#endif  // SAIR_MAPPED_DOMAIN_H_
