// RUN: sair-opt %s -convert-sair-to-loop

func @main(%arg0: memref<512x512xf32>, %arg1: memref<512x512xf32>, %arg2: memref<512x512xf32>) {
  %c0 = constant 0 : index
  %cst = constant 0.000000e+00 : f32

  sair.program  {
    %3 = sair.from_scalar %arg0 : !sair.value<(), memref<512x512xf32>>
    %4 = sair.from_scalar %arg1 : !sair.value<(), memref<512x512xf32>>
    %5 = sair.from_scalar %arg2 : !sair.value<(), memref<512x512xf32>>
    %6 = sair.static_range : !sair.static_range<512>
    %7 = sair.from_memref %3 memref[d0:%6, d1:%6] {buffer_name = "A"}
      : #sair.shape<d0:static_range<512> x d1:static_range<512>>, memref<512x512xf32>
    %8 = sair.from_memref %4 memref[d0:%6, d1:%6] {buffer_name = "B"}
      : #sair.shape<d0:static_range<512> x d1:static_range<512>>, memref<512x512xf32>
    %9 = sair.from_scalar %cst : !sair.value<(), f32>
    %10 = sair.copy[d0:%6, d1:%6] %9 {
      loop_nest = [
        {iter = #sair.mapping_expr<stripe(d0, [2])>, name = "loop_0"},
        {iter = #sair.mapping_expr<stripe(d1, [29])>, name = "loop_1"},
        {iter = #sair.mapping_expr<stripe(d0, [2, 1])>, name = "loop_2"},
        {iter = #sair.mapping_expr<stripe(d1, [29, 3])>, name = "loop_3"},
        {iter = #sair.mapping_expr<stripe(d1, [29, 3, 1])>, name = "loop_4"}
      ],
      storage = [{
        layout = #sair.named_mapping<
          [d0:"loop_0", d1:"loop_1", d2:"loop_2", d3:"loop_3", d4:"loop_4"]
            -> (unstripe(d0, d2, [2, 1]), unstripe(d1, d3, d4, [29, 3, 1]))>,
          name = "C", space = "memory"
      }]
    } : !sair.value<d0:static_range<512> x d1:static_range<512>, f32>
    %11 = sair.fby[d0:%6, d1:%6] %10(d0, d1) then[d2:%6] %12(d0, d1, d2)
      : !sair.value<d0:static_range<512> x d1:static_range<512> x d2:static_range<512>, f32>
    %12 = sair.map[d0:%6, d1:%6, d2:%6] %11(d0, d1, d2), %7(d0, d2), %8(d1, d2) attributes {
      loop_nest = [
        {iter = #sair.mapping_expr<stripe(d0, [3])>, name = "loop_5"},
        {iter = #sair.mapping_expr<stripe(d2, [3])>, name = "loop_6"},
        {iter = #sair.mapping_expr<stripe(d2, [3, 1])>, name = "loop_7"},
        {iter = #sair.mapping_expr<stripe(d0, [3, 1])>, name = "loop_8"},
        {iter = #sair.mapping_expr<stripe(d1, [3])>, name = "loop_9"},
        {iter = #sair.mapping_expr<stripe(d1, [3, 1])>, name = "loop_10"}
      ],
      storage = [{
        layout = #sair.named_mapping<
          [d0:"loop_5", d1:"loop_8", d2:"loop_9", d3:"loop_10"]
            -> (unstripe(d0, d1, [3, 1]), unstripe(d2, d3, [3, 1]))>,
        name = "C", space = "memory"}
      ]} {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: f32, %arg7: f32, %arg8: f32):  // no predecessors
      %14 = mulf %arg7, %arg8 : f32
      %15 = addf %arg6, %14 : f32
      sair.return %15 : f32
    } : #sair.shape<d0:static_range<512> x d1:static_range<512> x d2:static_range<512>>,
        (f32, f32, f32) -> f32
    %13 = sair.proj_last[d0:%6, d1:%6] of[d2:%6] %12(d0, d1, d2)
      : #sair.shape<d0:static_range<512> x d1:static_range<512> x d2:static_range<512>>, f32
    sair.to_memref %5 memref[d0:%6, d1:%6] %13(d0, d1) {buffer_name = "C"}
      : #sair.shape<d0:static_range<512> x d1:static_range<512>>, memref<512x512xf32>
    sair.exit
  }
  return
}
