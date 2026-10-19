"""Restyle frozen vector audit figures without rerunning their historical studies.

Optional pypdf dependency; this module is independent of the CSV-only renderer.
Open data paths, text, raster images, page bounds, and marker centers are preserved.
"""

from pathlib import Path


def restyle(source, destination, *, line_scale=0.6, marker_scale=0.6, inline_squares=False):
    from pypdf import PdfReader, PdfWriter
    from pypdf.generic import ContentStream, FloatObject

    reader = PdfReader(source)
    before_text = [p.extract_text() for p in reader.pages]
    writer = PdfWriter(clone_from=source)
    counts = {"stroke_commands": 0, "marker_paths": 0}
    visited = set()

    def transform(stream, *, marker_form=False):
        content = ContentStream(stream, writer)
        operations = content.operations
        path_start = None
        path_ops = {b"m", b"l", b"c", b"v", b"y", b"h", b"re"}
        for index, (args, op) in enumerate(operations):
            if op == b"w":
                args[0] = FloatObject(float(args[0]) * line_scale)
                counts["stroke_commands"] += 1
            if op in path_ops:
                if path_start is None:
                    path_start = index
                continue
            if path_start is None:
                continue
            if op not in {b"S", b"s", b"B", b"B*", b"b", b"b*", b"f", b"f*", b"n"}:
                continue
            segment = operations[path_start:index]
            codes = [code for _, code in segment]
            coordinates = [
                (values, j)
                for values, code in segment
                if code in {b"m", b"l", b"c", b"v", b"y"}
                for j in range(0, len(values), 2)
            ]
            circular = codes.count(b"c") >= 4
            cross = codes.count(b"m") == 2 and codes.count(b"l") == 2 and len(codes) == 4
            polygon = (marker_form or inline_squares) and b"h" in codes
            # Filled grid cells and ordinary open data paths are never marker candidates.
            if (
                coordinates
                and b"re" not in codes
                and (circular or cross or polygon)
                and op in {b"B", b"B*", b"b", b"b*", b"S", b"s"}
            ):
                xs = [float(values[j]) for values, j in coordinates]
                ys = [float(values[j + 1]) for values, j in coordinates]
                width, height = max(xs) - min(xs), max(ys) - min(ys)
                if 0 < width <= 12 and 0 < height <= 12 and 0.5 < width / height < 2:
                    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                    for values, j in coordinates:
                        values[j] = FloatObject(cx + marker_scale * (float(values[j]) - cx))
                        values[j + 1] = FloatObject(cy + marker_scale * (float(values[j + 1]) - cy))
                    counts["marker_paths"] += 1
            path_start = None
        return content

    def forms(resources):
        resources = resources.get_object() if hasattr(resources, "get_object") else resources
        for name, ref in resources.get("/XObject", {}).get_object().items() if resources.get("/XObject") else ():
            obj = ref.get_object()
            if obj.get("/Subtype") != "/Form" or id(obj) in visited:
                continue
            visited.add(id(obj))
            updated = transform(obj, marker_form=str(name).startswith("/M"))
            obj.set_data(updated.get_data())
            forms(obj.get("/Resources", {}))

    for page in writer.pages:
        forms(page.get("/Resources", {}))
        page.replace_contents(transform(page.get_contents()))
        page.compress_content_streams()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        writer.write(handle)
    check = PdfReader(destination)
    assert [p.extract_text() for p in check.pages] == before_text
    assert all(
        abs(float(a) - float(b)) < 1e-6
        for new, old in zip(check.pages, reader.pages)
        for a, b in zip(new.mediabox, old.mediabox)
    )
    return counts
