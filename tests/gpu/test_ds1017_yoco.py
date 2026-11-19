"""DeepInfer's row gather at decoder-boundary shapes on CUDA."""
import pytest
import torch

from kernels.ds1017_yoco import gather


@pytest.mark.parametrize("shape", [(33,), (33, 4), (33, 4, 5120), (7, 5120)])
@pytest.mark.parametrize("dtype", [torch.int32, torch.float32, torch.bfloat16])
def test_bounded_replay_gather_copies_rows_and_zeroes_placeholders(shape: tuple[int, ...],
                                                                 dtype: torch.dtype) -> None:
    tensor = torch.arange(int(torch.tensor(shape).prod()), device="cuda").reshape(shape).to(dtype)
    index = torch.tensor([0, shape[0] - 1, -1, shape[0], 2], dtype=torch.int32, device="cuda")
    want = torch.stack([tensor[0], tensor[-1], torch.zeros_like(tensor[0]),
                        torch.zeros_like(tensor[0]), tensor[2]])
    assert torch.equal(gather(tensor, index), want)
    if tensor.ndim > 1:
        view = tensor[..., :tensor.shape[-1] // 2]
        assert torch.equal(gather(view, index), want[..., :view.shape[-1]])


def test_prefill_window_metadata_matches_absolute_positions_for_ragged_rows() -> None:
    from kernels.ds1017_triton import prefill_window_meta

    counts, starts, rows = [145, 1, 7, 35], [0, 9, 32, 160], [6, 2, 5, 1]
    lower, decoding, ring_bases = [0, 0, 100, 157], [0, 1, 0, 0], [0, 0, 0, 128]
    bases: list[int] = []
    kept_bases: list[int] = []
    kept: list[int] = []
    expected: list[list[int]] = [[] for _ in range(5)]
    windows: list[list[int]] = []
    stores: list[int] = []
    for i, count in enumerate(counts):
        base = len(windows)
        bases.append(base)
        kept_bases.append(len(stores))
        kept.append(min(128, count))
        for j in range(count):
            pos = starts[i] + j
            for v, dst in zip((i, rows[i], starts[i], base, pos), expected, strict=True):
                dst.append(v)
            first = pos - 127 if decoding[i] else max(0, pos - 127, lower[i])
            windows.append([(-1 if key < 0 or key > pos else 256 + base + key - starts[i]
                             if key >= starts[i] else ring_bases[i] + key % 128) for key in range(first, first + 128)])
            if j >= count - 128:
                stores.append(base + j)
    meta = torch.tensor([rows, counts, starts, bases, kept, kept_bases, ring_bases, decoding, lower], device="cuda")
    coords, listed, keep = prefill_window_meta(meta, sum(counts), max(counts), len(stores), 256, 128)
    assert coords.tolist() == expected and listed.tolist() == windows and keep.tolist() == stores


def test_assign_pages_expands_ragged_page_ids_into_req_to_token_table() -> None:
    from array import array
    from kernels.page_assignment import assign_pages

    page = 64
    table = torch.zeros((8, 512), dtype=torch.int32, device="cuda")
    expected = table.clone()
    rows_pages = [(1, 0, [10, 25, 3]), (4, 2, [7, 19])]
    packet = array("i")
    flat: list[int] = []
    longest = 0
    for row, dst_page, pages in rows_pages:
        packet.extend((row, dst_page, len(flat), len(pages)))
        flat.extend(pages)
        longest = max(longest, len(pages))
        for j, p in enumerate(pages):
            start = (dst_page + j) * page
            expected[row, start:start + page] = torch.arange((p + 1) * page, (p + 2) * page,
                                                             dtype=torch.int32, device="cuda")
    nrows = len(rows_pages)
    packet.extend(flat)
    dev_pkt = torch.tensor(list(packet), dtype=torch.int32, device="cuda")
    assign_pages(table, dev_pkt, nrows, longest, page)
    assert torch.equal(table, expected)
