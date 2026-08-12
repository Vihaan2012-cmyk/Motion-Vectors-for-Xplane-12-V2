> **Correction — this result is void.**
>
> The 30/32 above was computed entirely from **magnitudes and ratios**, and the
> velocity field was **negated** at the time: the shader wrote `curr - prev`
> instead of `prev - curr`. A flipped vector has exactly the right magnitude, so
> every ratio on this page — 0.999, 1.001, 0.996, 1.003 — is a real number about
> the wrong quantity. The field pointed the opposite way in all 32 samples.
>
> The yardstick was also wrong. `far` was extracted from column 3 of the
> reprojection, which is the image of the camera's own origin, divided by a w row
> of -1/near = -61.9. The PITCH row's "0.918 and 1.090 fall outside the +-5% band"
> was that broken predictor, not the field.
>
> Superseded by 0.0.10, which measures a depth-free epipolar residual of
> 0.000-0.003 px and states what it still cannot account for.
