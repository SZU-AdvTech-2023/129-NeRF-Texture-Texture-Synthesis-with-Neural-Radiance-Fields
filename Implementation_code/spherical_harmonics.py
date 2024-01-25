from torch import Tensor

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def evaluate_spherical_harmonics(
    degree: int, sh_coeffs: Tensor, viewdirs: Tensor
) -> Tensor:
    assert 4 > degree >= 0, "only degrees 0, 1, 2, and 3 are supported :)"
    assert (degree + 1) ** 2 == sh_coeffs.shape[-1], (
        f"number of sh_coeffs ({sh_coeffs.shape[-1]}) do not match "
        f"the expected num_coeffs ({(degree + 1) ** 2}) for requested degree ({degree})"
    )

    result = C0 * sh_coeffs[..., 0]
    if degree > 0:
        x, y, z = viewdirs[..., 0:1], viewdirs[..., 1:2], viewdirs[..., 2:3]
        result = (
            result
            - C1 * y * sh_coeffs[..., 1]
            + C1 * z * sh_coeffs[..., 2]
            - C1 * x * sh_coeffs[..., 3]
        )
        if degree > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + C2[0] * xy * sh_coeffs[..., 4]
                + C2[1] * yz * sh_coeffs[..., 5]
                + C2[2] * (2.0 * zz - xx - yy) * sh_coeffs[..., 6]
                + C2[3] * xz * sh_coeffs[..., 7]
                + C2[4] * (xx - yy) * sh_coeffs[..., 8]
            )
            if degree > 2:
                result = (
                    result
                    + C3[0] * y * (3 * xx - yy) * sh_coeffs[..., 9]
                    + C3[1] * xy * z * sh_coeffs[..., 10]
                    + C3[2] * y * (4 * zz - xx - yy) * sh_coeffs[..., 11]
                    + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh_coeffs[..., 12]
                    + C3[4] * x * (4 * zz - xx - yy) * sh_coeffs[..., 13]
                    + C3[5] * z * (xx - yy) * sh_coeffs[..., 14]
                    + C3[6] * x * (xx - 3 * yy) * sh_coeffs[..., 15]
                )
                if degree > 3:
                    result = (
                        result
                        + C4[0] * xy * (xx - yy) * sh_coeffs[..., 16]
                        + C4[1] * yz * (3 * xx - yy) * sh_coeffs[..., 17]
                        + C4[2] * xy * (7 * zz - 1) * sh_coeffs[..., 18]
                        + C4[3] * yz * (7 * zz - 3) * sh_coeffs[..., 19]
                        + C4[4] * (zz * (35 * zz - 30) + 3) * sh_coeffs[..., 20]
                        + C4[5] * xz * (7 * zz - 3) * sh_coeffs[..., 21]
                        + C4[6] * (xx - yy) * (7 * zz - 1) * sh_coeffs[..., 22]
                        + C4[7] * xz * (xx - 3 * yy) * sh_coeffs[..., 23]
                        + C4[8]
                        * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
                        * sh_coeffs[..., 24]
                    )
    return result