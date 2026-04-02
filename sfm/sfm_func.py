import json
import os
import shutil
import zlib
from colorsys import hsv_to_rgb
from html import escape

import imageio.v2 as imageio
import numpy as np
import torch
# import pycolmap

from matching import match_images
from utils.basic import Print
from utils.to_pycolmap import batch_torch_matrix_to_pycolmap
from utils.geometry import compute_epipolar_errors, homo
from tqdm import tqdm


def _env_flag(name, default=False):
    val = os.environ.get(name, "1" if default else "0").strip().lower()
    return val in {"1", "true", "yes", "y", "on"}


def _visible_mean(values, mask):
    mask_f = mask.float()
    denom = mask_f.sum(0).clamp_min(1.0)
    return (values.float() * mask_f).sum(0) / denom


def _inv1p_score(values):
    return 1.0 / (1.0 + torch.clamp(values.float(), min=0.0))


def _to_rgb_u8(images_hw3):
    if not isinstance(images_hw3, torch.Tensor):
        images_hw3 = torch.as_tensor(images_hw3)
    arr = images_hw3.detach().cpu().float().clamp(0.0, 1.0).numpy()
    return np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)


def _dense_query_grid_xy(height, width, device):
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack((xx, yy), dim=-1).reshape(-1, 2)


def _in_bounds_xy(xy, width, height):
    finite = torch.isfinite(xy).all(dim=-1)
    return (
        finite
        & (xy[:, 0] >= 0.0)
        & (xy[:, 0] <= float(width - 1))
        & (xy[:, 1] >= 0.0)
        & (xy[:, 1] <= float(height - 1))
    )


def _xy_to_bgr_u8(xy, width, height):
    x = float(xy[0]) / max(float(width - 1), 1.0)
    y = float(xy[1]) / max(float(height - 1), 1.0)
    h = (x + 0.37 * y) % 1.0
    r, g, b = hsv_to_rgb(h, 1.0, 1.0)
    return (
        int(round(b * 255.0)),
        int(round(g * 255.0)),
        int(round(r * 255.0)),
    )


def _draw_matches_side_by_side_local(img_src_u8, img_tgt_u8, xy_src, xy_tgt, width, height, *, radius):
    import cv2

    out = np.concatenate([img_src_u8, img_tgt_u8], axis=1)
    src_width = img_src_u8.shape[1]
    num_points = int(min(xy_src.shape[0], xy_tgt.shape[0]))
    for idx in range(num_points):
        p_src = xy_src[idx]
        p_tgt = xy_tgt[idx]
        if not np.isfinite(p_src).all() or not np.isfinite(p_tgt).all():
            continue
        x0, y0 = int(round(float(p_src[0]))), int(round(float(p_src[1])))
        x1, y1 = int(round(float(p_tgt[0]))), int(round(float(p_tgt[1])))
        if not (0 <= x0 < width and 0 <= y0 < height and 0 <= x1 < width and 0 <= y1 < height):
            continue
        col = _xy_to_bgr_u8((x0, y0), width, height)
        cv2.circle(out, (x0, y0), radius, col, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (src_width + x1, y1), radius, col, -1, lineType=cv2.LINE_AA)
    return out


def _stack_pair_image_vertically(tile_rgb):
    h, w = tile_rgb.shape[:2]
    mid = w // 2
    top = tile_rgb[:, :mid]
    bottom = tile_rgb[:, mid:]
    out_w = max(top.shape[1], bottom.shape[1])
    if top.shape[1] != out_w:
        top = np.pad(top, ((0, 0), (0, out_w - top.shape[1]), (0, 0)), mode="constant")
    if bottom.shape[1] != out_w:
        bottom = np.pad(bottom, ((0, 0), (0, out_w - bottom.shape[1]), (0, 0)), mode="constant")
    return np.concatenate([top, bottom], axis=0)


def _blank_pair_tile(img_src_u8, img_tgt_u8):
    h_out = int(img_src_u8.shape[0] + img_tgt_u8.shape[0])
    w_out = int(max(img_src_u8.shape[1], img_tgt_u8.shape[1]))
    out = np.zeros((h_out, w_out, 3), dtype=np.uint8)
    out[: img_src_u8.shape[0], : img_src_u8.shape[1]] = img_src_u8
    out[img_src_u8.shape[0]: img_src_u8.shape[0] + img_tgt_u8.shape[0], : img_tgt_u8.shape[1]] = img_tgt_u8
    return out


def _draw_pair_stage_tile(img_src_u8, img_tgt_u8, xy_src, xy_tgt, width, height, *, radius=2):
    if xy_src.shape[0] == 0:
        return _blank_pair_tile(img_src_u8, img_tgt_u8)
    side_by_side = _draw_matches_side_by_side_local(
        img_src_u8,
        img_tgt_u8,
        xy_src,
        xy_tgt,
        width,
        height,
        radius=radius,
    )
    return _stack_pair_image_vertically(side_by_side)


def _draw_pair_stage_tile_colored(img_src_u8, img_tgt_u8, xy_src, xy_tgt, colors_bgr, *, radius=1):
    import cv2

    if xy_src.shape[0] == 0:
        return _blank_pair_tile(img_src_u8, img_tgt_u8)
    top_h, top_w = img_src_u8.shape[:2]
    bot_h, bot_w = img_tgt_u8.shape[:2]
    out_w = int(max(top_w, bot_w))
    out_h = int(top_h + bot_h)
    out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    out[:top_h, :top_w] = img_src_u8
    out[top_h:top_h + bot_h, :bot_w] = img_tgt_u8
    n = int(min(xy_src.shape[0], xy_tgt.shape[0], colors_bgr.shape[0]))
    for i in range(n):
        p0 = xy_src[i]
        p1 = xy_tgt[i]
        if not np.isfinite(p0).all() or not np.isfinite(p1).all():
            continue
        x0, y0 = int(round(float(p0[0]))), int(round(float(p0[1])))
        x1, y1 = int(round(float(p1[0]))), int(round(float(p1[1])))
        if not (0 <= x0 < top_w and 0 <= y0 < top_h and 0 <= x1 < bot_w and 0 <= y1 < bot_h):
            continue
        col = tuple(int(v) for v in colors_bgr[i])
        cv2.circle(out, (x0, y0), radius, col, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1, top_h + y1), radius, col, -1, lineType=cv2.LINE_AA)
    return out


def _skew_symmetric(t):
    tx = torch.zeros((3, 3), dtype=t.dtype, device=t.device)
    tx[0, 1] = -t[2]
    tx[0, 2] = t[1]
    tx[1, 0] = t[2]
    tx[1, 2] = -t[0]
    tx[2, 0] = -t[1]
    tx[2, 1] = t[0]
    return tx


def _fundamental_from_cameras(w2c_src, w2c_tgt, K_src, K_tgt):
    R_src, t_src = w2c_src[:3, :3], w2c_src[:3, 3]
    R_tgt, t_tgt = w2c_tgt[:3, :3], w2c_tgt[:3, 3]
    R_rel = R_tgt @ R_src.T
    t_rel = t_tgt - R_rel @ t_src
    E = _skew_symmetric(t_rel) @ R_rel
    K_src_inv = torch.linalg.inv(K_src)
    K_tgt_inv_t = torch.linalg.inv(K_tgt).T
    return K_tgt_inv_t @ E @ K_src_inv


def _sample_indices(num_points, max_points, seed):
    if num_points <= 0:
        return np.zeros((0,), dtype=np.int64)
    if num_points <= max_points:
        return np.arange(num_points, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_points, size=max_points, replace=False).astype(np.int64))


def _random_rainbow_bgr_u8(num_points, seed):
    if num_points <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    hues = np.linspace(0.0, 1.0, num_points, endpoint=False, dtype=np.float32)
    rng.shuffle(hues)
    colors = np.zeros((num_points, 3), dtype=np.uint8)
    for i, h in enumerate(hues):
        r, g, b = hsv_to_rgb(float(h), 0.9, 1.0)
        colors[i] = np.array([
            int(round(b * 255.0)),
            int(round(g * 255.0)),
            int(round(r * 255.0)),
        ], dtype=np.uint8)
    return colors


def _draw_epipolar_line(canvas, line_rgb, width, height, color_rgb, thickness=1):
    import cv2

    a, b, c = [float(v) for v in line_rgb]
    eps = 1e-8
    if abs(a) < eps and abs(b) < eps:
        return
    if abs(b) > abs(a):
        y0 = -c / (b + eps)
        y1 = -(a * float(width - 1) + c) / (b + eps)
        pt1 = (0, int(round(y0)))
        pt2 = (int(width - 1), int(round(y1)))
    else:
        x0 = -c / (a + eps)
        x1 = -(b * float(height - 1) + c) / (a + eps)
        pt1 = (int(round(x0)), 0)
        pt2 = (int(round(x1)), int(height - 1))
    ok, clipped_pt1, clipped_pt2 = cv2.clipLine((0, 0, int(width), int(height)), pt1, pt2)
    if not ok:
        return
    cv2.line(canvas, clipped_pt1, clipped_pt2, color_rgb, thickness, lineType=cv2.LINE_AA)


def _draw_pair_epiline_tile(
        img_src_u8,
        img_tgt_u8,
        xy_src,
        xy_tgt,
        F_src_to_tgt,
        *,
        max_draw=24,
        max_lines=24,
        seed=0,
        radius=1,
        epi_thresh_px=1.0,
):
    import cv2

    if xy_src.shape[0] == 0:
        return _blank_pair_tile(img_src_u8, img_tgt_u8), {
            "sampled": 0,
            "available": 0,
            "epi_inl": 0,
            "epi_out": 0,
            "epi_err_mean": 0.0,
        }

    top_h, top_w = img_src_u8.shape[:2]
    bot_h, bot_w = img_tgt_u8.shape[:2]
    out = _blank_pair_tile(img_src_u8, img_tgt_u8)
    top = out[:top_h, :top_w]
    bot = out[top_h:top_h + bot_h, :bot_w]
    sample_ids = _sample_indices(xy_src.shape[0], max_draw, seed)
    if sample_ids.size == 0:
        return out, {
            "sampled": 0,
            "available": int(xy_src.shape[0]),
            "epi_inl": 0,
            "epi_out": 0,
            "epi_err_mean": 0.0,
        }
    pts_src = xy_src[sample_ids]
    pts_tgt = xy_tgt[sample_ids]
    lines_tgt = homo(torch.from_numpy(pts_src).float()) @ F_src_to_tgt.detach().cpu().float().T
    lines_tgt = lines_tgt.numpy()
    lines_src = homo(torch.from_numpy(pts_tgt).float()) @ F_src_to_tgt.detach().cpu().float()
    lines_src = lines_src.numpy()
    tgt_h = homo(torch.from_numpy(pts_tgt).float()).numpy()
    den = np.linalg.norm(lines_tgt[:, :2], axis=1)
    epi_err = np.abs(np.sum(lines_tgt * tgt_h, axis=1)) / np.clip(den, 1e-8, None)
    epi_inlier = epi_err < float(epi_thresh_px)
    line_draw_ids = _sample_indices(sample_ids.size, max_lines, seed ^ 0xA5A5F00D)
    draw_line_mask = np.zeros((sample_ids.size,), dtype=bool)
    draw_line_mask[line_draw_ids] = True
    colors_bgr = _random_rainbow_bgr_u8(sample_ids.size, seed ^ 0x51A7C0DE)
    for idx in range(sample_ids.size):
        p0 = pts_src[idx]
        p1 = pts_tgt[idx]
        if not np.isfinite(p0).all() or not np.isfinite(p1).all():
            continue
        x0, y0 = int(round(float(p0[0]))), int(round(float(p0[1])))
        x1, y1 = int(round(float(p1[0]))), int(round(float(p1[1])))
        if not (0 <= x0 < top_w and 0 <= y0 < top_h and 0 <= x1 < bot_w and 0 <= y1 < bot_h):
            continue
        col = colors_bgr[idx]
        if draw_line_mask[idx]:
            _draw_epipolar_line(top, lines_src[idx], top_w, top_h, tuple(int(v) for v in col), thickness=1)
            _draw_epipolar_line(bot, lines_tgt[idx], bot_w, bot_h, tuple(int(v) for v in col), thickness=1)
        cv2.circle(top, (x0, y0), radius, tuple(int(v) for v in col), -1, lineType=cv2.LINE_AA)
        cv2.circle(bot, (x1, y1), radius, tuple(int(v) for v in col), -1, lineType=cv2.LINE_AA)
    return out, {
        "sampled": int(sample_ids.size),
        "available": int(xy_src.shape[0]),
        "line_drawn": int(draw_line_mask.sum()),
        "epi_inl": int(epi_inlier.sum()),
        "epi_out": int(sample_ids.size - int(epi_inlier.sum())),
        "epi_err_mean": float(epi_err.mean()) if epi_err.size else 0.0,
    }


def _write_scene_pair_html(scene_root, scene_name, ordered_rows):
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        f"<title>{escape(scene_name)}</title>",
        "<style>",
        "body{background:#0f1116;color:#e6e6e6;font:14px/1.4 -apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif;margin:24px;}",
        "a{color:#9ecbff;text-decoration:none}",
        "a:hover{text-decoration:underline}",
        "table{border-collapse:collapse;width:100%;}",
        "th,td{border:1px solid #2b2f3a;padding:10px;vertical-align:top;}",
        "th{position:sticky;top:0;background:#171b22;z-index:1;}",
        "tr:nth-child(even){background:#12161d;}",
        ".pair{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;white-space:nowrap;}",
        "img{display:block;width:100%;max-width:760px;height:auto;background:#000;}",
        ".cell{min-width:320px}",
        ".meta{color:#9da7b3;font-size:12px;margin:0 0 8px 0}",
        "</style></head><body>",
        f"<h1>{escape(scene_name)}</h1>",
        "<table>",
        "<p class='meta'>5_EPI colors the raw pair matches by BA-camera epipolar agreement: green=inlier, red=outlier.</p>",
        "<p class='meta'>6_EPI_LINES samples up to 24 3_BA_TRACKS correspondences, assigns each one a deterministic random rainbow color, and draws anti-aliased 1px BA-camera epipolar lines in both images for those same 24 points.</p>",
        "<thead><tr><th>Pair</th><th>1_RAW</th><th>2_BA_GATE</th><th>3_BA_TRACKS</th><th>4_DLT</th><th>5_EPI</th><th>6_EPI_LINES</th></tr></thead>",
        "<tbody>",
    ]
    for row in ordered_rows:
        lines.append("<tr>")
        lines.append(f"<td class='pair'>{escape(row['pair'])}</td>")
        for stage_name in ("1_RAW", "2_BA_GATE", "3_BA_TRACKS", "4_DLT", "5_EPI", "6_EPI_LINES"):
            rel = row["images"].get(stage_name)
            count = row["counts"].get(stage_name)
            meta = row.get("meta", {}).get(stage_name)
            if rel is None:
                lines.append("<td class='cell'></td>")
                continue
            rel_posix = rel.replace(os.sep, "/")
            if meta is None:
                meta = f"{stage_name}"
                if count is not None:
                    meta += f" | n={int(count)}"
            lines.append(
                "<td class='cell'>"
                f"<a href='{escape(rel_posix)}'><img loading='lazy' src='{escape(rel_posix)}' alt='{escape(stage_name)} {escape(row['pair'])}'></a>"
                f"<div class='meta'>{escape(meta)}</div>"
                "</td>"
            )
        lines.append("</tr>")
    lines.extend(["</tbody></table>", "</body></html>"])
    with open(os.path.join(scene_root, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _maybe_write_match_debug_grids(
        *,
        images_ff,
        pred_matches_lr,
        mask_ba_dense,
        mask_dlt_dense_post_epi,
        mask_epi_dense,
        tracks_ba,
        tracks_mask_ba,
        ba_intrinsics,
        ba_extrinsics,
        epi_thresh_px,
        output_dir,
):
    if not _env_flag("GGPTSFM_MATCH_GRIDS", default=False):
        return
    grid_root = output_dir or os.environ.get("GGPTSFM_MATCH_GRID_DIR", "").strip()
    if not grid_root:
        return

    images_u8 = _to_rgb_u8(images_ff)
    num_views, height, width = images_u8.shape[:3]
    query_xy = _dense_query_grid_xy(height, width, pred_matches_lr.device)
    stage_records = {"1_RAW": [], "2_BA_GATE": [], "3_BA_TRACKS": [], "4_DLT": [], "5_EPI": [], "6_EPI_LINES": []}

    for src in range(num_views):
        for tgt in range(src + 1, num_views):
            pair_name = f"{src:02d}-{tgt:02d}"
            xy_src_all = query_xy
            xy_tgt_all = pred_matches_lr[tgt, src].detach().float()
            raw_mask = _in_bounds_xy(xy_src_all, width, height) & _in_bounds_xy(xy_tgt_all, width, height)
            raw_idx = torch.nonzero(raw_mask, as_tuple=False).reshape(-1)
            raw_tile = _draw_pair_stage_tile(
                images_u8[src],
                images_u8[tgt],
                xy_src_all[raw_idx].detach().cpu().numpy(),
                xy_tgt_all[raw_idx].detach().cpu().numpy(),
                width,
                height,
                radius=1,
            )
            stage_records["1_RAW"].append({"pair": pair_name, "count": int(raw_mask.sum().item()), "tile": raw_tile})

            epi_dense_mask = raw_mask & mask_epi_dense[tgt, src].detach().bool()
            epi_idx = raw_idx
            epi_inlier_raw = epi_dense_mask[epi_idx].detach().cpu().numpy().astype(bool)
            epi_colors = np.zeros((epi_idx.shape[0], 3), dtype=np.uint8)
            epi_colors[epi_inlier_raw] = np.array([0, 255, 0], dtype=np.uint8)
            epi_colors[~epi_inlier_raw] = np.array([255, 0, 0], dtype=np.uint8)
            epi_tile = _draw_pair_stage_tile_colored(
                images_u8[src],
                images_u8[tgt],
                xy_src_all[epi_idx].detach().cpu().numpy(),
                xy_tgt_all[epi_idx].detach().cpu().numpy(),
                epi_colors,
                radius=1,
            )
            epi_inlier_count = int(epi_dense_mask.sum().item())
            epi_total_count = int(raw_mask.sum().item())
            stage_records["5_EPI"].append({
                "pair": pair_name,
                "count": epi_total_count,
                "tile": epi_tile,
                "meta": f"5_EPI | raw={epi_total_count} | epi_inl={epi_inlier_count} | epi_out={epi_total_count - epi_inlier_count}",
            })

            ba_dense_mask = raw_mask & mask_ba_dense[tgt, src].detach().bool()
            ba_dense_idx = torch.nonzero(ba_dense_mask, as_tuple=False).reshape(-1)
            ba_tile = _draw_pair_stage_tile(
                images_u8[src],
                images_u8[tgt],
                xy_src_all[ba_dense_idx].detach().cpu().numpy(),
                xy_tgt_all[ba_dense_idx].detach().cpu().numpy(),
                width,
                height,
                radius=1,
            )
            stage_records["2_BA_GATE"].append({"pair": pair_name, "count": int(ba_dense_mask.sum().item()), "tile": ba_tile})

            ba_track_mask = tracks_mask_ba[src].detach().bool() & tracks_mask_ba[tgt].detach().bool()
            if ba_track_mask.any():
                ba_track_xy_src = tracks_ba[src].detach().float()
                ba_track_xy_tgt = tracks_ba[tgt].detach().float()
                ba_track_mask = (
                    ba_track_mask
                    & _in_bounds_xy(ba_track_xy_src, width, height)
                    & _in_bounds_xy(ba_track_xy_tgt, width, height)
                )
                ba_track_idx = torch.nonzero(ba_track_mask, as_tuple=False).reshape(-1)
                ba_track_tile = _draw_pair_stage_tile(
                    images_u8[src],
                    images_u8[tgt],
                    ba_track_xy_src[ba_track_idx].detach().cpu().numpy(),
                    ba_track_xy_tgt[ba_track_idx].detach().cpu().numpy(),
                    width,
                    height,
                    radius=1,
                )
                ba_track_count = int(ba_track_mask.sum().item())
            else:
                ba_track_tile = _blank_pair_tile(images_u8[src], images_u8[tgt])
                ba_track_count = 0
            stage_records["3_BA_TRACKS"].append({"pair": pair_name, "count": ba_track_count, "tile": ba_track_tile})
            if ba_track_count > 0:
                F_src_to_tgt = _fundamental_from_cameras(
                    ba_extrinsics[src].detach().float(),
                    ba_extrinsics[tgt].detach().float(),
                    ba_intrinsics[src].detach().float(),
                    ba_intrinsics[tgt].detach().float(),
                )
                line_seed = zlib.crc32(f"{os.path.basename(grid_root)}|{pair_name}|6_EPI_LINES".encode("utf-8")) & 0xFFFFFFFF
                epi_line_tile, epi_line_stats = _draw_pair_epiline_tile(
                    images_u8[src],
                    images_u8[tgt],
                    ba_track_xy_src[ba_track_idx].detach().cpu().numpy(),
                    ba_track_xy_tgt[ba_track_idx].detach().cpu().numpy(),
                    F_src_to_tgt,
                    max_draw=24,
                    max_lines=24,
                    seed=line_seed,
                    radius=1,
                    epi_thresh_px=epi_thresh_px,
                )
                stage_records["6_EPI_LINES"].append({
                    "pair": pair_name,
                    "count": ba_track_count,
                    "tile": epi_line_tile,
                    "meta": (
                        f"6_EPI_LINES | sampled={epi_line_stats['sampled']}/{epi_line_stats['available']} "
                        f"| lines={epi_line_stats['line_drawn']} "
                        f"| epi_inl={epi_line_stats['epi_inl']} | epi_out={epi_line_stats['epi_out']} "
                        f"| mean_err={epi_line_stats['epi_err_mean']:.2f}px"
                    ),
                })
            else:
                stage_records["6_EPI_LINES"].append({
                    "pair": pair_name,
                    "count": 0,
                    "tile": _blank_pair_tile(images_u8[src], images_u8[tgt]),
                    "meta": "6_EPI_LINES | sampled=0/0 | lines=0 | epi_inl=0 | epi_out=0 | mean_err=0.00px",
                })

            dlt_dense_post_mask = raw_mask & mask_dlt_dense_post_epi[tgt, src].detach().bool()
            dlt_dense_post_idx = torch.nonzero(dlt_dense_post_mask, as_tuple=False).reshape(-1)
            dlt_tile = _draw_pair_stage_tile(
                images_u8[src],
                images_u8[tgt],
                xy_src_all[dlt_dense_post_idx].detach().cpu().numpy(),
                xy_tgt_all[dlt_dense_post_idx].detach().cpu().numpy(),
                width,
                height,
                radius=1,
            )
            stage_records["4_DLT"].append({"pair": pair_name, "count": int(dlt_dense_post_mask.sum().item()), "tile": dlt_tile})

    if os.path.isdir(grid_root):
        shutil.rmtree(grid_root)
    os.makedirs(grid_root, exist_ok=True)
    manifest = {"num_views": int(num_views), "pairs": []}
    row_map = {}
    for stage_name in ("1_RAW", "2_BA_GATE", "3_BA_TRACKS", "4_DLT", "5_EPI", "6_EPI_LINES"):
        stage_dir = os.path.join(grid_root, stage_name)
        os.makedirs(stage_dir, exist_ok=True)
        for record in stage_records[stage_name]:
            pair_name = record["pair"]
            filename = f"{pair_name}.png"
            out_path = os.path.join(stage_dir, filename)
            imageio.imwrite(out_path, record["tile"])
            row = row_map.setdefault(pair_name, {"pair": pair_name, "images": {}, "counts": {}, "meta": {}})
            row["images"][stage_name] = os.path.join(stage_name, filename)
            row["counts"][stage_name] = int(record["count"])
            if "meta" in record:
                row["meta"][stage_name] = str(record["meta"])
    ordered_rows = [row_map[key] for key in sorted(row_map.keys())]
    manifest["pairs"] = ordered_rows
    with open(os.path.join(grid_root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    _write_scene_pair_html(grid_root, os.path.basename(grid_root.rstrip(os.sep)), ordered_rows)


def run_sfm(
        images,
        ff_outputs,
        match_models,
        cfg,
        gt=None,
        output_dir=None,
):
    import pycolmap
    images_ff = ff_outputs['images_ff']  # B,H,W,3
    N = images_ff.shape[0]
    device = images_ff.device
    ff_h, ff_w = images_ff.shape[1:3]
    output_dict = {} 

    """
    1. Dense matching
    """
    match_on_ff_res = os.environ.get("GGPT_MATCH_ON_FF_RES", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
    if match_on_ff_res:
        images_for_matching = images_ff
        Print("GGPT_MATCH_ON_FF_RES=1: running matcher on feedforward-resolution images.")
    else:
        if isinstance(images, torch.Tensor)==False:
            # first convert to torch tensor
            images = torch.from_numpy(np.stack([np.array(img) for img in images], axis=0)).float()/255.0  #(N,H,W,3)
            images = images.to(device)
        images_for_matching = images
    m_h, m_w = images_for_matching.shape[1:3]
    sx, sy = ff_w/m_w, ff_h/m_h
    mres_to_fres = torch.tensor([[sx,0,0.5*(sx-1)],[0,sy,0.5*(sy-1)],[0,0,1]], dtype=torch.float32).to(device)  #3,3
    match_results = match_images(
        match_models=match_models,
        images_hr=images_for_matching.permute(0,3,1,2),  #N,3,H,W
        lr_h=ff_h, lr_w=ff_w,
        hr_to_lr=mres_to_fres
    )
    pred_scores = match_results['pred_scores']
    pred_cycle_error = match_results['pred_cycle_error']
    pred_scores_flat = pred_scores.reshape(N, -1)
    pred_cycle_error_flat = pred_cycle_error.reshape(N, -1)

    M_ba = (pred_scores > cfg.ba_config.score_thresh) & (pred_cycle_error < cfg.ba_config.cycle_err_thresh)
    M_dlt = (pred_scores > cfg.dlt_config.score_thresh) & (pred_cycle_error < cfg.dlt_config.cycle_err_thresh)
    M_ba_dense = M_ba.clone()
    M_dlt_dense_pre_epi = M_dlt.clone()
    M_epi_dense = torch.zeros_like(M_ba_dense, dtype=torch.bool)

    """
    2. Sparse ba (TODO: support known camera poses)
    """
    # 2.1 Chooes tracks for BA
    M_ba = M_ba.reshape(N,-1) #Ntgt, Nsrc*H*W
    query_sp_scores = match_results['sp_scores'].view(-1)  #Nsrc*H*W
    tracknum_perview = torch.zeros(N, dtype=torch.int32).to(device)
    selected = torch.zeros(M_ba.shape[1], dtype=torch.bool).to(device) #Num_tracks=Nsrc*H*W
    assert M_ba.shape[1] == (N*ff_h*ff_w)
    for ni in tqdm(range(N), desc="Selecting tracks for BA"):
        to_select_num = cfg.ba_config.mintrack_per_view - tracknum_perview[ni]
        if to_select_num<=0:
            continue
        candidate_tracks = M_ba[ni]&(M_ba.sum(axis=0)>=2)&(~selected) # visible in this view, at least visible in 2 views, not selected, (Ntracks,)
        candidate_ids = torch.where(candidate_tracks)[0]
        if len(candidate_ids)==0:
            continue
        candidate_sps = query_sp_scores[candidate_ids]
        selected_ids = candidate_ids[torch.argsort(candidate_sps, descending=True)][:to_select_num]
        selected[selected_ids] = True
        tracknum_perview += M_ba[:,selected_ids].sum(axis=1).to(torch.int32)
    Print(f"The number of tracks covering each view: {[nn.item() for nn in tracknum_perview]}")
    tracks_ba = match_results['pred_matches_lr'].view(N,-1,2)[:,selected]  #N, Ntracks_ba, 2
    tracks_mask_ba = M_ba[:,selected]  #Ntgt, Ntracks_ba
    tracks_score_ba = pred_scores_flat[:,selected]
    tracks_cycle_ba = pred_cycle_error_flat[:,selected]
    if tracks_mask_ba.shape[1] == 0:
        raise AssertionError(
            "No BA tracks selected after score/cycle filtering; "
            f"views={N}, ff_hw=({ff_h},{ff_w}), "
            f"tracks_visible_ge_2={(M_ba.sum(axis=0) >= 2).sum().item()}, "
            f"mintrack_per_view={cfg.ba_config.mintrack_per_view}"
        )
    assert tracks_mask_ba.sum(axis=0).min()>=2, "Each track should be visible in at least two views for BA."
    pts3d_ba = ff_outputs['points'].reshape(-1,3)[selected]  #Ntracks_ba, 3 (Takes the ff's prediction at query positions as initialization)

    if cfg.match_config.save_vis:
        from matching.vis_match import vis_matches
        vis_matches(images=images_ff.permute(0,3,1,2),  #N,3,H,W 
            matches=tracks_ba,
            visibility=tracks_mask_ba, # visualize the matches with queries from image i
            filename=os.path.join(output_dir, f'matches_for_ba.png'), 
            vis_mask=tracks_mask_ba.sum(axis=0)>=2, # visualize len>=2 tracks (Ntgt,Num_tracks) (Actually it is all zero)
            vis_num_track=10)

    calibrated = cfg.ba_config.get('calibrated', False)
    ba_intrinsics = gt['intrinsics'][:,:3,:3].to(device).float() if calibrated else ff_outputs['intrinsics'][:,:3,:3]
    reconstruction = batch_torch_matrix_to_pycolmap(
        points3d = pts3d_ba,
        tracks = tracks_ba+0.5, masks = tracks_mask_ba,
        extrinsics = ff_outputs['extrinsics'][:,:3,:4],
        intrinsics = ba_intrinsics+0.5,
        image_size = [ff_w, ff_h],
        camera_type = cfg.ba_config.camera_type,
        shared_camera = cfg.ba_config.shared_camera,
    )
    refine_focal = not calibrated and cfg.ba_config.get('refine_focal_length', True)
    # refine_pp = not calibrated
    if cfg.ba_config.loss_function_type == 'cauchy':
        loss_function_type = pycolmap.LossFunctionType.CAUCHY
    else:
        loss_function_type = pycolmap.LossFunctionType.TRIVIAL
    ba_options = pycolmap.BundleAdjustmentOptions(
        loss_function_type=loss_function_type, loss_function_scale=cfg.ba_config.loss_function_scale,
        refine_focal_length=refine_focal)
    ba_options.solver_options.minimizer_progress_to_stdout = bool(int(os.environ.get("GGPT_BA_PROGRESS", "0")))
    ba_config = pycolmap.BundleAdjustmentConfig()
    for img_id in reconstruction.images.keys():
        ba_config.add_image(img_id)
    bundle_adjuster = pycolmap.create_default_bundle_adjuster(ba_options, ba_config, reconstruction)
    bundle_adjuster.solve()
    reconstruction.update_point_3d_errors()

    output_dict['intrinsics'] = torch.zeros_like(ff_outputs['intrinsics'])
    output_dict['extrinsics'] = torch.zeros_like(ff_outputs['extrinsics'])
    exp_cx_colmap = ff_w / 2.0
    exp_cy_colmap = ff_h / 2.0
    colmap_pp_tol = 1e-3
    for i in range(1,N+1): # Image ids in pycolmap start from 1
        camera = reconstruction.cameras[reconstruction.images[i].camera_id]
        cam_params = camera.params
        if camera.model == pycolmap.CameraModelId.SIMPLE_PINHOLE:
            fx = float(cam_params[0])
            fy = fx
            cx_colmap = float(cam_params[1])
            cy_colmap = float(cam_params[2])
        elif camera.model == pycolmap.CameraModelId.PINHOLE:
            fx = float(cam_params[0])
            fy = float(cam_params[1])
            cx_colmap = float(cam_params[2])
            cy_colmap = float(cam_params[3])
        else:
            raise AssertionError(f"Unsupported GGPT BA camera model: {camera.model}, params={cam_params}")
        if abs(cx_colmap - exp_cx_colmap) > colmap_pp_tol or abs(cy_colmap - exp_cy_colmap) > colmap_pp_tol:
            raise AssertionError(
                "GGPT BA returned a non-centered principal point, but this export path assumes fixed centered PP. "
                f"camera_params={cam_params}, expected_colmap_pp=({exp_cx_colmap:.6f},{exp_cy_colmap:.6f}), "
                f"got=({cx_colmap:.6f},{cy_colmap:.6f})"
            )
        output_dict['intrinsics'][i-1, 0, 0] = fx
        output_dict['intrinsics'][i-1, 1, 1] = fy
        output_dict['intrinsics'][i-1, 0, 2] = cx_colmap - 0.5
        output_dict['intrinsics'][i-1, 1, 2] = cy_colmap - 0.5
        output_dict['intrinsics'][i-1, 2, 2] = 1.0
        rigid3d = reconstruction.images[i].cam_from_world.matrix() # (3,4)
        output_dict['extrinsics'][i-1,:3,:] = torch.from_numpy(rigid3d).to(device).float()
        if output_dict['extrinsics'].shape[1] == 4:
            output_dict['extrinsics'][i-1,3,3] = 1.0
    output_dict['camera_success'] = True

    sfm_points_world = []
    sparse_score_variants = {"match_score_mean": [], "cycle_inv1p": [], "reproj_inv1p": []}
    sparse_intr = output_dict['intrinsics'][:, :3, :3].to(device).float()
    sparse_extr = output_dict['extrinsics'][:, :3, :4].to(device).float()
    tracks_ba_obs = tracks_ba.to(device).float() + 0.5
    match_score_mean_ba = _visible_mean(tracks_score_ba, tracks_mask_ba)
    cycle_score_ba = _inv1p_score(_visible_mean(tracks_cycle_ba, tracks_mask_ba))
    for pid in sorted(reconstruction.points3D.keys()):
        track_idx = int(pid) - 1
        if track_idx < 0 or track_idx >= tracks_ba.shape[1]:
            raise AssertionError(
                f"Unexpected BA point3D id={pid} for {tracks_ba.shape[1]} selected tracks."
            )
        p3d = reconstruction.points3D[pid]
        xyz = np.asarray(p3d.xyz, dtype=np.float32)
        if np.isfinite(xyz).all() is False:
            continue
        xyz_t = torch.from_numpy(xyz).to(device=device, dtype=torch.float32)
        xyz_h = torch.cat([xyz_t, xyz_t.new_tensor([1.0])], dim=0)
        pts_cam = torch.einsum('nij,j->ni', sparse_extr, xyz_h)
        uv_h = torch.einsum('nij,nj->ni', sparse_intr, pts_cam)
        uv = uv_h[:, :2] / uv_h[:, 2:3].clamp_min(1e-8)
        reproj_valid = tracks_mask_ba[:, track_idx] & torch.isfinite(uv).all(dim=1) & (pts_cam[:, 2] > 1e-8)
        if bool(reproj_valid.any()):
            reproj_err = torch.norm(uv[reproj_valid] - tracks_ba_obs[reproj_valid, track_idx], dim=-1).mean()
            reproj_inv1p = float(_inv1p_score(reproj_err))
        else:
            reproj_inv1p = 0.0
        match_score_mean = float(match_score_mean_ba[track_idx])
        cycle_inv1p = float(cycle_score_ba[track_idx])
        sfm_points_world.append(xyz)
        sparse_score_variants["match_score_mean"].append(match_score_mean)
        sparse_score_variants["cycle_inv1p"].append(cycle_inv1p)
        sparse_score_variants["reproj_inv1p"].append(reproj_inv1p)
    if len(sfm_points_world) > 0:
        output_dict["sfm_points_world"] = torch.from_numpy(np.asarray(sfm_points_world, dtype=np.float32)).to(device)
        output_dict["sfm_points_score"] = torch.from_numpy(
            np.asarray(sparse_score_variants["reproj_inv1p"], dtype=np.float32)
        ).to(device)
        output_dict["sfm_points_score_variants"] = {
            key: torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device)
            for key, values in sparse_score_variants.items()
        }
    else:
        output_dict["sfm_points_world"] = torch.zeros((0, 3), dtype=torch.float32, device=device)
        output_dict["sfm_points_score"] = torch.zeros((0,), dtype=torch.float32, device=device)
        output_dict["sfm_points_score_variants"] = {
            "match_score_mean": torch.zeros((0,), dtype=torch.float32, device=device),
            "cycle_inv1p": torch.zeros((0,), dtype=torch.float32, device=device),
            "reproj_inv1p": torch.zeros((0,), dtype=torch.float32, device=device),
        }


    """
    3. Direct linear Triangulation
    """
    intrinsic_dlt, extrinsic_dlt = output_dict['intrinsics'].to(device), output_dict['extrinsics'].to(device) 
    P = torch.einsum('nij,njk->nik', intrinsic_dlt, extrinsic_dlt[:,:3,:4]).float() #N,3,4
    camR, camT = extrinsic_dlt[:,:3,:3], extrinsic_dlt[:,:3,-1]
    camC = -torch.einsum('nij,nj->ni', camR.permute(0,2,1), camT) #(N,3)
    # filter matches with epipolar error (TODO to make it more efficient)
    w2c_s, K_s = output_dict['extrinsics'].to(device), output_dict['intrinsics'].to(device)
    M_dlt = M_dlt.to(device)
    dlt_num = (M_dlt.sum(axis=0)>=2).sum().item()

    M_ba_debug = M_ba.reshape(N,N,-1).clone()
    for ni in range(N):
        w2c_0, K_0 = output_dict['extrinsics'][ni], output_dict['intrinsics'][ni]
        matches = match_results['pred_matches_lr'][:,ni].view(N,ff_h,ff_w,2) #(N, H,W, 2)
        dis_a, dis_b = compute_epipolar_errors(w2c_0, w2c_s, K_0, K_s, matches.view(N,ff_h,ff_w,2)) #(N,H,W)
        epipolar_errors_msk = (dis_a < cfg.dlt_config.max_epipolar_error) & (dis_b < cfg.dlt_config.max_epipolar_error) 
        epipolar_errors_msk[ni,:,:] = True  #self-view
        epipolar_errors_msk = epipolar_errors_msk.view(N, ff_h*ff_w) #(Ntgt, H*W)
        M_epi_dense[:, ni] = epipolar_errors_msk
        M_dlt[:,ni] = M_dlt[:,ni] & epipolar_errors_msk  #(Ntgt, H*W)
        M_ba_debug[:,ni] = M_ba_debug[:,ni] & epipolar_errors_msk  #(Ntgt, H*W)
    M_dlt_dense_post_epi = M_dlt.clone()
    M_dlt = M_dlt.reshape(N,-1) #Ntgt, Nsrc*H*W
    remaining_tracks = (M_dlt.sum(axis=0)>=2)  #(Nsrc*H*W,)
    print(f"Number of tracks used for DLT triangulation: {remaining_tracks.sum().item()}")
    print("Epipolar error discard ratio: ", 1-(M_dlt.sum(axis=0)>=2).sum().item()/dlt_num)
    tracks_dlt = match_results['pred_matches_lr'].view(N,-1,2)[:,remaining_tracks].to(device)  #Nview, Ntracks_dlt, 2
    tracks_mask_dlt = M_dlt[:,remaining_tracks]  #Nview, Ntracks_dlt
    tracks_score_dlt = pred_scores_flat[:,remaining_tracks].to(device)
    tracks_cycle_dlt = pred_cycle_error_flat[:,remaining_tracks].to(device)

    _maybe_write_match_debug_grids(
        images_ff=images_ff,
        pred_matches_lr=match_results['pred_matches_lr'],
        mask_ba_dense=M_ba_dense,
        mask_dlt_dense_post_epi=M_dlt_dense_post_epi,
        mask_epi_dense=M_epi_dense,
        tracks_ba=tracks_ba,
        tracks_mask_ba=tracks_mask_ba,
        ba_intrinsics=output_dict['intrinsics'],
        ba_extrinsics=output_dict['extrinsics'],
        epi_thresh_px=cfg.dlt_config.max_epipolar_error,
        output_dir=output_dir,
    )

    weights = tracks_mask_dlt.float()
    max_pts_num = cfg.dlt_config.get('batch_size', 500000)
    num_chunk = (tracks_dlt.shape[1]+max_pts_num-1)//max_pts_num
    xyz_in_img, count_in_img = torch.zeros(N*ff_h*ff_w, 3).to(device), torch.zeros(N*ff_h*ff_w).to(device) # Our target
    dlt_score_sum = {
        "match_score_mean": torch.zeros(N*ff_h*ff_w, dtype=torch.float32, device=device),
        "cycle_inv1p": torch.zeros(N*ff_h*ff_w, dtype=torch.float32, device=device),
        "reproj_inv1p": torch.zeros(N*ff_h*ff_w, dtype=torch.float32, device=device),
    }
    for chunk_id in tqdm(range(num_chunk), desc='DLT triangulation'):
        start_idx = chunk_id*max_pts_num
        end_idx = min((chunk_id+1)*max_pts_num, tracks_dlt.shape[1])
        tracks_dlt_chunk = tracks_dlt[:,start_idx:end_idx]  #(Nview, Ntracks_chunk, 2)
        tracks_mask_dlt_chunk = tracks_mask_dlt[:,start_idx:end_idx]  #(Nview, Ntracks_chunk)
        tracks_score_dlt_chunk = tracks_score_dlt[:,start_idx:end_idx]
        tracks_cycle_dlt_chunk = tracks_cycle_dlt[:,start_idx:end_idx]
        
        Ai_chunk = tracks_dlt_chunk[...,None] * P[:,None,2:3,:] - P[:,None,:2,:] #(Nview,Ntracks,2,4)
        weights_chunk = weights[:,start_idx:end_idx] #(Nview, Ntracks_chunk)
        AitAi = Ai_chunk.permute(0,1,3,2) @ Ai_chunk #(Nview,Ntracks_chunk,4,4) 
        AitAi = AitAi * weights_chunk[:,:,None,None] #(Nview,Ntracks_chunk,4,4)
        AitAi_sum = AitAi.sum(axis=0) #(Ntracks_chunk,4,4)
        _, eigenvectors_chunk = torch.linalg.eigh(AitAi_sum) #(Ntracks_chunk,4), (Ntracks_chunk,4,4)
        pt3d_chunk = eigenvectors_chunk[:,:,0] #(Ntracks_chunk,4) Ascending order, the first column vector
        
        # Filter points with invalid solutions
        xyz = pt3d_chunk[:,:3] / pt3d_chunk[:,3:4] #(Npts,3)
        filter1 = pt3d_chunk[:,3].abs()>1e-10  #valid solution
        xyz = xyz[filter1]
        tracks_dlt_chunk = tracks_dlt_chunk[:,filter1]
        tracks_mask_dlt_chunk = tracks_mask_dlt_chunk[:,filter1]
        tracks_score_dlt_chunk = tracks_score_dlt_chunk[:,filter1]
        tracks_cycle_dlt_chunk = tracks_cycle_dlt_chunk[:,filter1]

        # Filter points based on reprojection error
        xy_reproj_chunk = torch.einsum('nij,mj->nmi', P, homo(xyz))
        xy_reproj_chunk = xy_reproj_chunk[:,:,:2]/xy_reproj_chunk[:,:,2:3]
        reproj_error_chunk = torch.norm(xy_reproj_chunk-tracks_dlt_chunk, dim=-1)
        reproj_error_mean_chunk = (reproj_error_chunk*tracks_mask_dlt_chunk).sum(0)/tracks_mask_dlt_chunk.sum(0)  
        filter_reproj = (reproj_error_mean_chunk < cfg.dlt_config.max_reproj_error)
        xyz = xyz[filter_reproj]
        tracks_dlt_chunk = tracks_dlt_chunk[:,filter_reproj]
        tracks_mask_dlt_chunk = tracks_mask_dlt_chunk[:,filter_reproj]
        tracks_score_dlt_chunk = tracks_score_dlt_chunk[:,filter_reproj]
        tracks_cycle_dlt_chunk = tracks_cycle_dlt_chunk[:,filter_reproj]
        reproj_error_mean_chunk = reproj_error_mean_chunk[filter_reproj]
        if filter_reproj.sum()==0:
            continue
        # Filter points based on triangulation angles
        # We need further batch operations here. quadratic to the number of views
        max_pts_num2 = int(min(max_pts_num//N, tracks_dlt_chunk.shape[1]))
        num_chunk2 = (tracks_dlt_chunk.shape[1]+max_pts_num2-1)//max_pts_num2
        for chunk_id2 in range(num_chunk2):
            start_idx2 = chunk_id2*max_pts_num2 
            end_idx2 = min((chunk_id2+1)*max_pts_num2, tracks_dlt_chunk.shape[1])
            rays_chunk = xyz[None,start_idx2:end_idx2,...] - camC[:,None,:] #(N,Ntracks_chunk,3)
            rays_chunk = rays_chunk / torch.linalg.norm(rays_chunk, axis=-1, keepdim=True) #(N,Ntracks_chunk,3)
            cos_angles = (rays_chunk[None,:,:,:]*rays_chunk[:,None,:,:]).sum(-1) #(N,N,Ntracks_chunk)-> (Nview1, Nview2, Ntracks_chunk)
            angle_radians = torch.acos(torch.clamp(cos_angles, -0.9999, 0.9999))
            angle_degs = torch.rad2deg(angle_radians)
            vismask_pairwise = tracks_mask_dlt_chunk[None,:, start_idx2:end_idx2] & tracks_mask_dlt_chunk[:,None,start_idx2:end_idx2]  #(Nview1, Nview2, Ntracks_chunk)
            angle_degs = angle_degs * vismask_pairwise.float() # Set invisible views to 0 angle 
            max_angle_degs_chunk = angle_degs.view(-1, end_idx2-start_idx2).max(0).values #(Ntracks_chunk,)
            if chunk_id2==0:
                max_angle_degs = max_angle_degs_chunk
            else:
                max_angle_degs = torch.cat([max_angle_degs, max_angle_degs_chunk], 0)
        filter_angle = (max_angle_degs > cfg.dlt_config.min_tri_angle)
        xyz = xyz[filter_angle]
        tracks_dlt_chunk = tracks_dlt_chunk[:,filter_angle]
        tracks_mask_dlt_chunk = tracks_mask_dlt_chunk[:,filter_angle]
        tracks_score_dlt_chunk = tracks_score_dlt_chunk[:,filter_angle]
        tracks_cycle_dlt_chunk = tracks_cycle_dlt_chunk[:,filter_angle]
        reproj_error_mean_chunk = reproj_error_mean_chunk[filter_angle]
        max_angle_degs = max_angle_degs[filter_angle]

        match_score_mean_chunk = _visible_mean(tracks_score_dlt_chunk, tracks_mask_dlt_chunk)
        cycle_score_chunk = _inv1p_score(_visible_mean(tracks_cycle_dlt_chunk, tracks_mask_dlt_chunk))
        reproj_inv1p_chunk = _inv1p_score(reproj_error_mean_chunk)

        xyz_for_img_chunk = xyz.unsqueeze(0).tile(N,1,1)[tracks_mask_dlt_chunk] #(N,Npts_chunk,3) -> (Nobs_chunk, 3)
        view_index_chunk = torch.arange(N).to(device).unsqueeze(1).tile(1,xyz.shape[0])[tracks_mask_dlt_chunk]  #(N,Npts) -> (Nobs_chunk,)
        index2d_obs_chunk = tracks_dlt_chunk[tracks_mask_dlt_chunk]
        index2d_in_img_chunk = index2d_obs_chunk.round().long()
        valid_obs = torch.isfinite(index2d_obs_chunk).all(dim=1)
        valid_obs = valid_obs & (index2d_in_img_chunk[:, 0] >= 0) & (index2d_in_img_chunk[:, 0] < ff_w)
        valid_obs = valid_obs & (index2d_in_img_chunk[:, 1] >= 0) & (index2d_in_img_chunk[:, 1] < ff_h)
        if valid_obs.sum() == 0:
            continue
        xyz_for_img_chunk = xyz_for_img_chunk[valid_obs]
        view_index_chunk = view_index_chunk[valid_obs]
        index2d_in_img_chunk = index2d_in_img_chunk[valid_obs]
        index1d_in_scene_chunk = view_index_chunk*ff_w*ff_h + index2d_in_img_chunk[:,1]*ff_w + index2d_in_img_chunk[:,0]  #(Nobs_chunk,)
        xyz_in_img.scatter_reduce_(dim=0, index=index1d_in_scene_chunk.unsqueeze(1).expand(-1,3), src=xyz_for_img_chunk, reduce='sum', include_self=True)
        count_in_img.scatter_reduce_(dim=0, index=index1d_in_scene_chunk, src=torch.ones_like(index1d_in_scene_chunk).float(), reduce='sum', include_self=True)
        track_scores_for_obs = {
            "match_score_mean": match_score_mean_chunk.unsqueeze(0).expand(N, -1)[tracks_mask_dlt_chunk][valid_obs],
            "cycle_inv1p": cycle_score_chunk.unsqueeze(0).expand(N, -1)[tracks_mask_dlt_chunk][valid_obs],
            "reproj_inv1p": reproj_inv1p_chunk.unsqueeze(0).expand(N, -1)[tracks_mask_dlt_chunk][valid_obs],
        }
        for key, obs_score in track_scores_for_obs.items():
            dlt_score_sum[key].scatter_reduce_(
                dim=0,
                index=index1d_in_scene_chunk,
                src=obs_score.float(),
                reduce='sum',
                include_self=True,
            )
    
    xyz_in_img = xyz_in_img / count_in_img.clamp(min=1)[:,None]
    dlt_xyz, count_in_img = xyz_in_img.view(N,ff_h,ff_w,3), count_in_img.view(N,ff_h,ff_w)
    dlt_mask = (count_in_img>0)  #(N,H,W)

    if dlt_mask.sum()==0:
        print('No valid DLT points after assigning to image pixels')
        output_dict['points_success'] = False
        return output_dict

    output_dict['points'] = dlt_xyz
    output_dict['point_masks'] = dlt_mask
    output_dict['point_scores'] = (dlt_score_sum["reproj_inv1p"] / count_in_img.view(-1).clamp(min=1.0)).view(N,ff_h,ff_w)
    output_dict['point_score_variants'] = {
        key: (value / count_in_img.view(-1).clamp(min=1.0)).view(N,ff_h,ff_w)
        for key, value in dlt_score_sum.items()
    }
    output_dict['points_success'] = True

    return output_dict
    
