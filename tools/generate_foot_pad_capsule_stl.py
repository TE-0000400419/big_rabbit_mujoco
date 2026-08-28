#!/usr/bin/env python3
"""Generate printable half-capsule foot-pad shell STL files from MuJoCo geoms.

The outer curved half exactly follows the collision capsule.  The attachment
plane is left open, and an inward-offset half capsule plus a flat rim form a
watertight shell.  Mesh coordinates are written in the unit implied by the
matching ``<mesh scale=...>`` asset (mm in the current model), while the capsule
coordinates themselves are in metres.
"""

from __future__ import annotations

import argparse
import math
import struct
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Sequence


Vec3 = tuple[float, float, float]
Triangle = tuple[Vec3, Vec3, Vec3]


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = REPO_ROOT / "robotmodel" / "big_rabbit" / "big_rabbit.xml"


def add(a: Vec3, b: Vec3) -> Vec3:
    return tuple(a[index] + b[index] for index in range(3))  # type: ignore[return-value]


def sub(a: Vec3, b: Vec3) -> Vec3:
    return tuple(a[index] - b[index] for index in range(3))  # type: ignore[return-value]


def mul(a: Vec3, scalar: float) -> Vec3:
    return tuple(value * scalar for value in a)  # type: ignore[return-value]


def dot(a: Vec3, b: Vec3) -> float:
    return sum(a[index] * b[index] for index in range(3))


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


def normalized(a: Vec3) -> Vec3:
    length = norm(a)
    if length <= 1.0e-12:
        raise ValueError("capsule の fromto の両端が同じです")
    return mul(a, 1.0 / length)


def quaternion_rotate(quaternion: tuple[float, ...], vector: Vec3) -> Vec3:
    """Rotate vector by a MuJoCo quaternion in (w, x, y, z) order."""
    length = math.sqrt(sum(value * value for value in quaternion))
    if length <= 1.0e-12:
        raise ValueError("姿勢 quaternion がゼロです")
    w, x, y, z = (value / length for value in quaternion)
    imaginary: Vec3 = (x, y, z)
    first_cross = cross(imaginary, vector)
    second_cross = cross(imaginary, first_cross)
    return add(vector, add(mul(first_cross, 2.0 * w), mul(second_cross, 2.0)))


def parse_floats(text: str, expected: int, attribute: str) -> tuple[float, ...]:
    values = tuple(float(value) for value in text.split())
    if len(values) != expected:
        raise ValueError(f"{attribute} は {expected} 要素必要です: {text!r}")
    return values


def half_capsule_surface(
    point_a: Vec3,
    point_b: Vec3,
    radius: float,
    down_direction: Vec3,
    radial_segments: int,
    hemisphere_segments: int,
) -> tuple[list[Vec3], list[tuple[int, int, int]], list[int], Vec3]:
    """Return an open half-capsule curved surface and its cut-plane boundary."""
    if radius <= 0.0:
        raise ValueError("capsule 半径は正でなければなりません")
    if radial_segments < 8 or hemisphere_segments < 2:
        raise ValueError("分割数が小さすぎます")

    axis = normalized(sub(point_b, point_a))
    # The site quaternion preserves the original sole box orientation.  Remove
    # tiny numerical components parallel to the capsule axis before meshing.
    down = normalized(sub(down_direction, mul(axis, dot(down_direction, axis))))
    lateral = normalized(cross(down, axis))

    vertices: list[Vec3] = [sub(point_a, mul(axis, radius))]
    rings: list[list[int]] = []

    def append_ring(center: Vec3, ring_radius: float) -> None:
        ring = []
        # Only the down-facing semicircle is exposed.  Both endpoints lie on
        # the flat attachment plane and are shared with its closing face.
        for segment in range(radial_segments + 1):
            angle = -math.pi / 2.0 + math.pi * segment / radial_segments
            radial = add(mul(down, math.cos(angle)), mul(lateral, math.sin(angle)))
            ring.append(len(vertices))
            vertices.append(add(center, mul(radial, ring_radius)))
        rings.append(ring)

    # Hemisphere at point_a, from the -axis pole to its equator.
    for index in range(1, hemisphere_segments + 1):
        latitude = -math.pi / 2.0 + index * math.pi / (2.0 * hemisphere_segments)
        append_ring(
            add(point_a, mul(axis, radius * math.sin(latitude))),
            radius * math.cos(latitude),
        )

    # Cylinder end at point_b followed by the second hemisphere rings.
    append_ring(point_b, radius)
    for index in range(1, hemisphere_segments):
        latitude = index * math.pi / (2.0 * hemisphere_segments)
        append_ring(
            add(point_b, mul(axis, radius * math.sin(latitude))),
            radius * math.cos(latitude),
        )

    pole_b = len(vertices)
    vertices.append(add(point_b, mul(axis, radius)))

    faces: list[tuple[int, int, int]] = []
    first_ring = rings[0]
    for segment in range(radial_segments):
        faces.append((0, first_ring[segment + 1], first_ring[segment]))

    for lower, upper in zip(rings, rings[1:]):
        for segment in range(radial_segments):
            faces.append((lower[segment], lower[segment + 1], upper[segment + 1]))
            faces.append((lower[segment], upper[segment + 1], upper[segment]))

    last_ring = rings[-1]
    for segment in range(radial_segments):
        faces.append((last_ring[segment], last_ring[segment + 1], pole_b))

    # Ensure every curved triangle points away from the capsule axis segment.
    axis_vector = sub(point_b, point_a)
    axis_length_squared = dot(axis_vector, axis_vector)
    for index, face in enumerate(faces):
        triangle = tuple(vertices[vertex] for vertex in face)
        normal = cross(sub(triangle[1], triangle[0]), sub(triangle[2], triangle[0]))
        centroid = mul(add(add(triangle[0], triangle[1]), triangle[2]), 1.0 / 3.0)
        fraction = dot(sub(centroid, point_a), axis_vector) / axis_length_squared
        fraction = min(1.0, max(0.0, fraction))
        closest = add(point_a, mul(axis_vector, fraction))
        if dot(normal, sub(centroid, closest)) < 0.0:
            faces[index] = (face[0], face[2], face[1])

    # The boundary is a 2-D capsule on the attachment plane.  A shell connects
    # this loop to the corresponding loop of its inner offset surface.
    boundary = (
        [0]
        + [ring[0] for ring in rings]
        + [pole_b]
        + [ring[-1] for ring in reversed(rings)]
    )
    validate_surface(vertices, point_a, point_b, radius)
    for vertex in vertices:
        if dot(sub(vertex, point_a), down) < -1.0e-9:
            raise ValueError("平面より上側に頂点があります")
    return vertices, faces, boundary, down


def half_capsule_shell_mesh(
    point_a: Vec3,
    point_b: Vec3,
    radius: float,
    wall_thickness: float,
    down_direction: Vec3,
    radial_segments: int,
    hemisphere_segments: int,
) -> tuple[list[Vec3], list[tuple[int, int, int]]]:
    """Return a watertight, open-at-the-top half-capsule shell."""
    if wall_thickness <= 0.0 or wall_thickness >= radius:
        raise ValueError(
            f"肉厚は 0 より大きく半径未満でなければなりません: "
            f"thickness={wall_thickness}, radius={radius}"
        )

    outer_vertices, outer_faces, outer_boundary, down = half_capsule_surface(
        point_a,
        point_b,
        radius,
        down_direction,
        radial_segments,
        hemisphere_segments,
    )
    inner_radius = radius - wall_thickness
    inner_vertices, inner_faces, inner_boundary, _ = half_capsule_surface(
        point_a,
        point_b,
        inner_radius,
        down_direction,
        radial_segments,
        hemisphere_segments,
    )
    if len(outer_vertices) != len(inner_vertices) or len(outer_boundary) != len(inner_boundary):
        raise ValueError("外面と内面のトポロジーが一致しません")

    inner_offset = len(outer_vertices)
    vertices = outer_vertices + inner_vertices
    # Outer normals point away from the capsule.  Inner normals must point
    # into the cavity, so reverse their winding.
    faces = list(outer_faces)
    faces.extend(
        (face[0] + inner_offset, face[2] + inner_offset, face[1] + inner_offset)
        for face in inner_faces
    )

    # Bridge the two cut-plane loops.  This annular face is the 5 mm-wide rim
    # around the opening; its outward normal points opposite the solid's down.
    flat_outward = mul(down, -1.0)
    for index, outer_start in enumerate(outer_boundary):
        next_index = (index + 1) % len(outer_boundary)
        outer_end = outer_boundary[next_index]
        inner_start = inner_boundary[index] + inner_offset
        inner_end = inner_boundary[next_index] + inner_offset
        rim_faces = [
            (outer_start, outer_end, inner_end),
            (outer_start, inner_end, inner_start),
        ]
        for face in rim_faces:
            triangle = tuple(vertices[vertex] for vertex in face)
            normal = cross(sub(triangle[1], triangle[0]), sub(triangle[2], triangle[0]))
            if dot(normal, flat_outward) < 0.0:
                face = (face[0], face[2], face[1])
            faces.append(face)

    max_thickness_error = max(
        abs(norm(sub(outer, inner)) - wall_thickness)
        for outer, inner in zip(outer_vertices, inner_vertices)
    )
    if max_thickness_error > 1.0e-9:
        raise ValueError(f"肉厚が一定ではありません: error={max_thickness_error:.3e} m")
    validate_topology(vertices, faces)
    validate_winding(vertices, faces)
    return vertices, faces


def validate_topology(vertices: Sequence[Vec3], faces: Sequence[tuple[int, int, int]]) -> None:
    edges: dict[tuple[int, int], int] = {}
    edge_directions: dict[tuple[int, int], int] = {}
    for face in faces:
        if len(set(face)) != 3:
            raise ValueError(f"縮退した三角形があります: {face}")
        for start, end in zip(face, (face[1], face[2], face[0])):
            edge = tuple(sorted((start, end)))
            edges[edge] = edges.get(edge, 0) + 1
            edge_directions[edge] = edge_directions.get(edge, 0) + (1 if (start, end) == edge else -1)
    invalid = [edge for edge, count in edges.items() if count != 2]
    if invalid:
        raise ValueError(f"STL が閉曲面ではありません: invalid_edges={len(invalid)}")
    inconsistent = [edge for edge, direction in edge_directions.items() if direction != 0]
    if inconsistent:
        raise ValueError(f"隣接面の向きが一致しません: invalid_edges={len(inconsistent)}")
    if max(index for face in faces for index in face) >= len(vertices):
        raise ValueError("三角形が存在しない頂点を参照しています")


def validate_winding(vertices: Sequence[Vec3], faces: Sequence[tuple[int, int, int]]) -> None:
    signed_volume = sum(
        dot(vertices[face[0]], cross(vertices[face[1]], vertices[face[2]])) / 6.0
        for face in faces
    )
    if signed_volume <= 0.0:
        raise ValueError(f"面の向きが反転しています: signed_volume={signed_volume:.6e} m^3")


def validate_surface(vertices: Sequence[Vec3], point_a: Vec3, point_b: Vec3, radius: float) -> None:
    axis_vector = sub(point_b, point_a)
    axis_length_squared = dot(axis_vector, axis_vector)
    max_error = 0.0
    for vertex in vertices:
        fraction = dot(sub(vertex, point_a), axis_vector) / axis_length_squared
        fraction = min(1.0, max(0.0, fraction))
        closest = add(point_a, mul(axis_vector, fraction))
        max_error = max(max_error, abs(norm(sub(vertex, closest)) - radius))
    if max_error > 1.0e-9:
        raise ValueError(f"capsule 表面から外れた頂点があります: {max_error:.3e} m")


def triangles(
    vertices: Sequence[Vec3], faces: Iterable[tuple[int, int, int]], scale: Vec3
) -> list[Triangle]:
    result = []
    for face in faces:
        result.append(
            tuple(
                tuple(vertices[index][axis] / scale[axis] for axis in range(3))
                for index in face
            )
        )
    return result  # type: ignore[return-value]


def binary_stl(name: str, mesh_triangles: Sequence[Triangle]) -> bytes:
    header = f"big_rabbit {name} half capsule shell".encode("ascii")[:80].ljust(80, b"\0")
    payload = bytearray(header + struct.pack("<I", len(mesh_triangles)))
    for triangle in mesh_triangles:
        edge_a = sub(triangle[1], triangle[0])
        edge_b = sub(triangle[2], triangle[0])
        normal = normalized(cross(edge_a, edge_b))
        payload += struct.pack("<3f", *normal)
        for vertex in triangle:
            payload += struct.pack("<3f", *vertex)
        payload += struct.pack("<H", 0)
    return bytes(payload)


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as output:
        temporary = Path(output.name)
        output.write(payload)
    temporary.chmod(0o644)
    temporary.replace(path)


def generate(
    xml_path: Path,
    side: str,
    wall_thickness: float,
    radial_segments: int,
    hemisphere_segments: int,
) -> None:
    root = ET.parse(xml_path).getroot()
    compiler = root.find("compiler")
    mesh_dir = xml_path.parent / (compiler.get("meshdir", ".") if compiler is not None else ".")

    collision_name = f"{side}_foot_collision"
    visual_name = f"{side}_foot_pad_visual"
    contact_name = f"{side}_foot_contact"
    collision = root.find(f".//geom[@name='{collision_name}']")
    visual = root.find(f".//geom[@name='{visual_name}']")
    contact = root.find(f".//site[@name='{contact_name}']")
    if collision is None or collision.get("type") != "capsule":
        raise ValueError(f"capsule geom が見つかりません: {collision_name}")
    if visual is None or visual.get("type") != "mesh":
        raise ValueError(f"visual mesh geom が見つかりません: {visual_name}")
    if contact is None or not contact.get("quat"):
        raise ValueError(f"足裏の切断方向を表す site が見つかりません: {contact_name}")

    mesh_name = visual.get("mesh")
    asset = root.find(f"./asset/mesh[@name='{mesh_name}']")
    if asset is None or not asset.get("file"):
        raise ValueError(f"mesh asset が見つかりません: {mesh_name}")

    fromto = parse_floats(collision.get("fromto", ""), 6, f"{collision_name}.fromto")
    size = parse_floats(collision.get("size", ""), 1, f"{collision_name}.size")
    contact_quat = parse_floats(contact.get("quat", ""), 4, f"{contact_name}.quat")
    scale_values = parse_floats(asset.get("scale", "1 1 1"), 3, f"{mesh_name}.scale")
    if any(value <= 0.0 for value in scale_values):
        raise ValueError(f"mesh scale は正でなければなりません: {scale_values}")

    point_a: Vec3 = (fromto[0], fromto[1], fromto[2])
    point_b: Vec3 = (fromto[3], fromto[4], fromto[5])
    scale: Vec3 = (scale_values[0], scale_values[1], scale_values[2])
    down = quaternion_rotate(contact_quat, (0.0, 0.0, -1.0))
    vertices, faces = half_capsule_shell_mesh(
        point_a,
        point_b,
        size[0],
        wall_thickness,
        down,
        radial_segments,
        hemisphere_segments,
    )
    mesh_triangles = triangles(vertices, faces, scale)
    output_path = mesh_dir / asset.get("file")
    write_atomic(output_path, binary_stl(mesh_name or side, mesh_triangles))

    axis_length = norm(sub(point_b, point_a))
    radial_chord_error = size[0] * (1.0 - math.cos(math.pi / (2.0 * radial_segments)))
    hemisphere_chord_error = size[0] * (
        1.0 - math.cos(math.pi / (4.0 * hemisphere_segments))
    )
    chord_error = max(radial_chord_error, hemisphere_chord_error)
    print(
        f"{output_path}: triangles={len(faces)}, radius={size[0] * 1000:.1f} mm, "
        f"inner_radius={(size[0] - wall_thickness) * 1000:.1f} mm, "
        f"wall={wall_thickness * 1000:.1f} mm, "
        f"axis={axis_length * 1000:.1f} mm, total={((axis_length + 2.0 * size[0]) * 1000):.1f} mm, "
        f"open=attachment_plane, max_chord_error<={chord_error * 1000:.3f} mm"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--wall-thickness-mm", type=float, default=5.0)
    parser.add_argument("--radial-segments", type=int, default=32)
    parser.add_argument("--hemisphere-segments", type=int, default=16)
    args = parser.parse_args()
    for side in ("left", "right"):
        generate(
            args.xml.resolve(),
            side,
            args.wall_thickness_mm * 0.001,
            args.radial_segments,
            args.hemisphere_segments,
        )


if __name__ == "__main__":
    main()
