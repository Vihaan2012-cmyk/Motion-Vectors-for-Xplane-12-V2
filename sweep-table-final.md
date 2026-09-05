# MotionVectors stability sweep

Generated 9/5/2026 9:10:52 PM. 200s per run, 200-frame windows, third window reported. Mask: `taa.metrics_mask=2,taa.metrics_far=0.0001`.

**Baseline instability 0.2425, noise floor 0.0018** (spread over 3 identical runs). 
A row is only WORSE or better if its delta clears the floor; otherwise it is `-` and the sweep will not pretend to rank it.

INSTAB = 2000*static + 1000*edge + 100*flicker. Lower is more stable. Weights are a judgement; every component is in the table so you can rank on one instead.

| config | phase | instab | delta | verdict | static | edge | flicker | trms | lap | sharp | stair | novec | vel | sign |
|---|---|---:|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| taa.dlssd=1 (RR, placeholder guides) | dlssd | 4.2288 | +3.9863 | WORSE | 0.001053 | 0.000184 | 0.019382 | 0.008881 | 0.00838 | 0.01230 | 0.730 | 0.0975 | 6.24 | plus |
| taa.contact=0.6 + taa.sharpen=0.5 | pair | 1.0931 | +0.8506 | WORSE | 0.000381 | 0.000042 | 0.002904 | 0.007360 | 0.01899 | 0.01757 | 0.803 | 0.0953 | 6.10 | plus |
| taa.contact=0.6 + taa.gi=1 | pair | 0.8146 | +0.5721 | WORSE | 0.000301 | 0.000036 | 0.001760 | 0.007365 | 0.01218 | 0.01500 | 0.761 | 0.0937 | 6.00 | plus |
| taa.contact=0.6 + taa.unjitter=0 | pair | 0.7874 | +0.5449 | WORSE | 0.000294 | 0.000037 | 0.001626 | 0.007097 | 0.01342 | 0.01532 | 0.770 | 0.0951 | 6.09 | plus |
| taa.contact=0.6 + taa.nearfield_m=0 | pair | 0.7810 | +0.5385 | WORSE | 0.000287 | 0.000034 | 0.001729 | 0.007241 | 0.01196 | 0.01480 | 0.761 | 0.1030 | 6.59 | plus |
| taa.sharpen=0.5 + taa.unjitter=0 | pair | 0.7665 | +0.5240 | WORSE | 0.000218 | 0.000022 | 0.003086 | 0.000766 | 0.01980 | 0.01752 | 0.797 | 0.0969 | 6.20 | plus |
| taa.contact=0.6 | single | 0.7663 | +0.5238 | WORSE | 0.000286 | 0.000034 | 0.001597 | 0.007271 | 0.01190 | 0.01469 | 0.759 | 0.1121 | 7.17 | plus |
| taa.contact=0.6 + taa.cr_unjitter=0 | pair | 0.6691 | +0.4266 | WORSE | 0.000259 | 0.000029 | 0.001220 | 0.007276 | 0.00972 | 0.01347 | 0.738 | 0.1126 | 7.21 | plus |
| taa.sharpen=0.5 + taa.gi=1 | pair | 0.5398 | +0.2973 | WORSE | 0.000172 | 0.000014 | 0.001809 | 0.001456 | 0.01794 | 0.01710 | 0.791 | 0.1031 | 6.60 | plus |
| taa.sharpen=0.5 + taa.nearfield_m=0 | pair | 0.4832 | +0.2407 | WORSE | 0.000153 | 0.000013 | 0.001637 | 0.000640 | 0.01801 | 0.01723 | 0.792 | 0.0967 | 6.19 | plus |
| taa.sharpen=0.5 + taa.box_mod=0 | pair | 0.4830 | +0.2405 | WORSE | 0.000152 | 0.000014 | 0.001649 | 0.000576 | 0.01773 | 0.01707 | 0.786 | 0.1024 | 6.55 | plus |
| taa.sharpen=0.5 | single | 0.4690 | +0.2265 | WORSE | 0.000150 | 0.000013 | 0.001563 | 0.000621 | 0.01761 | 0.01684 | 0.791 | 0.1098 | 7.02 | plus |
| taa.sharpen=0.5 + taa.cr_unjitter=0 | pair | 0.3710 | +0.1285 | WORSE | 0.000113 | 0.000011 | 0.001334 | 0.000470 | 0.01552 | 0.01575 | 0.779 | 0.0965 | 6.17 | plus |
| taa.unjitter=0 + taa.gi=1 | pair | 0.3514 | +0.1089 | WORSE | 0.000117 | 0.000011 | 0.001061 | 0.001227 | 0.01256 | 0.01503 | 0.749 | 0.0900 | 5.76 | plus |
| taa.contact=0.6 + taa.box_mod=0 | pair | 0.3424 | +0.0999 | WORSE | 0.000121 | 0.000010 | 0.000905 | 0.002982 | 0.01103 | 0.01443 | 0.732 | 0.1075 | 6.88 | plus |
| taa.unjitter=0 + taa.nearfield_m=0 | pair | 0.3424 | +0.0999 | WORSE | 0.000109 | 0.000012 | 0.001131 | 0.000665 | 0.01203 | 0.01457 | 0.745 | 0.1137 | 7.29 | plus |
| taa.gi=1 + taa.nearfield_m=0 | pair | 0.3314 | +0.0889 | WORSE | 0.000099 | 0.000013 | 0.001218 | 0.001627 | 0.01117 | 0.01459 | 0.740 | 0.0954 | 6.11 | plus |
| taa.unjitter=0 | single | 0.3069 | +0.0644 | WORSE | 0.000102 | 0.000010 | 0.000926 | 0.000261 | 0.01246 | 0.01497 | 0.749 | 0.0924 | 5.91 | plus |
| taa.unjitter=0 + taa.box_mod=0 | pair | 0.3058 | +0.0633 | WORSE | 0.000102 | 0.000010 | 0.000919 | 0.000255 | 0.01240 | 0.01490 | 0.748 | 0.0981 | 6.28 | plus |
| taa.unjitter=0 + taa.cr_unjitter=0 | pair | 0.3038 | +0.0613 | WORSE | 0.000101 | 0.000010 | 0.000912 | 0.000253 | 0.01230 | 0.01477 | 0.748 | 0.0999 | 6.39 | plus |
| taa.gi=1 | single | 0.2928 | +0.0503 | WORSE | 0.000094 | 0.000010 | 0.000942 | 0.001382 | 0.01123 | 0.01460 | 0.741 | 0.0977 | 6.25 | plus |
| taa.nearfield_m=0 | single | 0.2813 | +0.0388 | WORSE | 0.000082 | 0.000012 | 0.001048 | 0.001153 | 0.01114 | 0.01462 | 0.740 | 0.0913 | 5.86 | plus |
| taa.ao=0.5 | single | 0.2443 | +0.0018 | - | 0.000078 | 0.000009 | 0.000796 | 0.000543 | 0.01120 | 0.01460 | 0.742 | 0.0935 | 5.98 | plus |
| baseline#3 | baseline | 0.2436 | +0.0011 | - | 0.000077 | 0.000009 | 0.000795 | 0.000531 | 0.01122 | 0.01463 | 0.740 | 0.0942 | 6.03 | plus |
| taa.dilate=0 | single | 0.2435 | +0.0010 | - | 0.000077 | 0.000009 | 0.000793 | 0.000568 | 0.01119 | 0.01454 | 0.734 | 0.0992 | 6.35 | plus |
| taa.fg=0 | single | 0.2429 | +0.0004 | - | 0.000077 | 0.000009 | 0.000795 | 0.000531 | 0.01114 | 0.01453 | 0.741 | 0.0993 | 6.35 | plus |
| taa.clear_after_resolve=1 | single | 0.2426 | +0.0001 | - | 0.000077 | 0.000009 | 0.000787 | 0.000534 | 0.01101 | 0.01436 | 0.741 | 0.1114 | 7.13 | plus |
| taa.engine_depth=1 | single | 0.2425 | +0.0000 | - | 0.000077 | 0.000009 | 0.000791 | 0.000528 | 0.01119 | 0.01459 | 0.740 | 0.0969 | 6.20 | plus |
| taa.reactive=1 | single | 0.2422 | -0.0003 | - | 0.000077 | 0.000009 | 0.000791 | 0.000531 | 0.01105 | 0.01442 | 0.742 | 0.1046 | 6.69 | plus |
| baseline#1 | baseline | 0.2421 | -0.0004 | - | 0.000077 | 0.000009 | 0.000786 | 0.000529 | 0.01087 | 0.01419 | 0.741 | 0.1192 | 7.63 | plus |
| taa.hist_catmull=0 | single | 0.2419 | -0.0006 | - | 0.000077 | 0.000009 | 0.000788 | 0.000527 | 0.01108 | 0.01447 | 0.740 | 0.1010 | 6.46 | plus |
| baseline#2 | baseline | 0.2418 | -0.0007 | - | 0.000077 | 0.000009 | 0.000783 | 0.000531 | 0.01096 | 0.01430 | 0.741 | 0.1143 | 7.31 | plus |
| taa.novec_reproject=0 | single | 0.2413 | -0.0012 | - | 0.000077 | 0.000009 | 0.000782 | 0.000527 | 0.01105 | 0.01442 | 0.742 | 0.1036 | 6.63 | plus |
| taa.taau=1 | single | 0.2410 | -0.0015 | - | 0.000077 | 0.000009 | 0.000779 | 0.000532 | 0.01101 | 0.01436 | 0.741 | 0.1106 | 7.08 | plus |
| taa.nearfield_m=0 + taa.box_mod=0 | pair | 0.2379 | -0.0046 | better | 0.000075 | 0.000008 | 0.000802 | 0.000539 | 0.01087 | 0.01461 | 0.725 | 0.0948 | 6.07 | plus |
| taa.gi=1 + taa.box_mod=0 | pair | 0.2091 | -0.0334 | better | 0.000072 | 0.000006 | 0.000584 | 0.000280 | 0.01069 | 0.01425 | 0.726 | 0.1178 | 7.54 | plus |
| taa.box_mod=0 | single | 0.2021 | -0.0404 | better | 0.000070 | 0.000006 | 0.000559 | 0.000270 | 0.01066 | 0.01424 | 0.726 | 0.1192 | 7.63 | plus |
| taa.gi=1 + taa.cr_unjitter=0 | pair | 0.1926 | -0.0499 | better | 0.000066 | 0.000006 | 0.000557 | 0.001266 | 0.00904 | 0.01341 | 0.711 | 0.0954 | 6.10 | plus |
| taa.nearfield_m=0 + taa.cr_unjitter=0 | pair | 0.1789 | -0.0636 | better | 0.000054 | 0.000006 | 0.000649 | 0.000493 | 0.00871 | 0.01307 | 0.709 | 0.1176 | 7.53 | plus |
| taa.cr_unjitter=0 | single | 0.1446 | -0.0979 | better | 0.000049 | 0.000005 | 0.000413 | 0.000226 | 0.00898 | 0.01338 | 0.710 | 0.0964 | 6.17 | plus |
| taa.box_mod=0 + taa.cr_unjitter=0 | pair | 0.1437 | -0.0988 | better | 0.000049 | 0.000005 | 0.000411 | 0.000221 | 0.00895 | 0.01333 | 0.709 | 0.0972 | 6.22 | plus |

**Least stable single flip: `taa.contact=0.6`** at 0.7663, +0.5238 over baseline.

Movers (cleared the floor): `taa.contact=0.6`, `taa.sharpen=0.5`, `taa.unjitter=0`, `taa.gi=1`, `taa.nearfield_m=0`, `taa.box_mod=0`, `taa.cr_unjitter=0`

**Least stable pair: `taa.contact=0.6 + taa.sharpen=0.5`** at 1.0931.

Columns: static = mean |delta| on unmoved pixels (pure instability); edge = |delta| weighted by local contrast (what the eye catches); flicker = fraction of pixels reversing delta sign at a 2-LSB threshold; trms = RMS residual; lap = Laplacian energy (aliasing proxy, higher = more high-frequency detail and more jaggies); sharp = gradient energy (a config cannot win by going soft); stair = intra-quad / neighbourhood variance; novec = fraction of pixels with no velocity written; vel = mean vector length in px; sign = which velocity convention reprojected better.

Limits: parked cockpit autoload, so ghost / disocclusion / sign are uninformative (no motion). Reverse-Z was measured, not assumed (54.6% of pixels below 0.0001).
