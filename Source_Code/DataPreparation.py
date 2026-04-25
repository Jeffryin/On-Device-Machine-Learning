from __future__ import annotations
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set
from pathlib import Path


@dataclass
class BoundingBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def validate(self) -> None:
        if self.xmax <= self.xmin:
            raise ValueError(f"Invalid box: xmax ({self.xmax}) <= xmin ({self.xmin})")
        if self.ymax <= self.ymin:
            raise ValueError(f"Invalid box: ymax ({self.ymax}) <= ymin ({self.ymin})")


@dataclass
class VocObject:
    class_name: str
    bbox: BoundingBox
    difficult: int = 0


@dataclass
class VocAnnotation:
    filename: str
    width: int
    height: int
    objects: List[VocObject]


class VocParser:
    def parse(self, xml_path: Path) -> VocAnnotation:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = self._get_text(root, "filename")
        size = root.find("size")
        if size is None:
            raise ValueError(f"Missing <size> in {xml_path}")

        width = int(self._get_text(size, "width"))
        height = int(self._get_text(size, "height"))

        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size in {xml_path}: {width}x{height}")

        objects: List[VocObject] = []

        for obj in root.findall("object"):
            class_name = self._get_text(obj, "name")
            difficult_text = obj.findtext("difficult", default="0").strip()
            difficult = int(difficult_text)

            bndbox = obj.find("bndbox")
            if bndbox is None:
                raise ValueError(f"Missing <bndbox> in {xml_path}")

            bbox = BoundingBox(
                xmin=float(self._get_text(bndbox, "xmin")),
                ymin=float(self._get_text(bndbox, "ymin")),
                xmax=float(self._get_text(bndbox, "xmax")),
                ymax=float(self._get_text(bndbox, "ymax")),
            )
            bbox.validate()

            objects.append(VocObject(class_name=class_name, bbox=bbox, difficult=difficult))

        return VocAnnotation(
            filename=filename,
            width=width,
            height=height,
            objects=objects,
        )

    @staticmethod
    def _get_text(parent: ET.Element, tag: str) -> str:
        node = parent.find(tag)
        if node is None or node.text is None:
            raise ValueError(f"Missing <{tag}>")
        return node.text.strip()


class ClassMapBuilder:
    def __init__(self, parser: VocParser) -> None:
        self.parser = parser

    def build_from_directory(self, annotation_dir: Path) -> Dict[str, int]:
        class_names: Set[str] = set()

        # Added filtering to ignore hidden files like .DS_Store or ._metadata
        xml_files = [f for f in annotation_dir.glob("*.xml") if not f.name.startswith('.')]

        for xml_file in sorted(xml_files):
            try:
                annotation = self.parser.parse(xml_file)
                for obj in annotation.objects:
                    class_names.add(obj.class_name)
            except ET.ParseError as e:
                print(f"Skipping {xml_file}: XML is not well-formed. Error: {e}")
                continue
            except Exception as e:
                print(f"Skipping {xml_file} due to unexpected error: {e}")
                continue

        sorted_names = sorted(
            class_names, 
            key=lambda x: (0, int(x)) if x.isdigit() else (1, x)
        )
        return {name: idx for idx, name in enumerate(sorted_names)}


class YoloConverter:
    def __init__(self, class_to_id: Dict[str, int]) -> None:
        self.class_to_id = class_to_id

    def convert_object(self, obj: VocObject, img_width: int, img_height: int) -> str:
        if obj.class_name not in self.class_to_id:
            raise KeyError(f"Unknown class '{obj.class_name}'")

        x_center, y_center, width, height = self._voc_to_yolo(
            obj.bbox,
            img_width,
            img_height,
        )

        class_id = self.class_to_id[obj.class_name]
        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    @staticmethod
    def _voc_to_yolo(
        bbox: BoundingBox,
        img_width: int,
        img_height: int,
    ) -> Tuple[float, float, float, float]:
        # Clamp to valid image bounds
        xmin = max(0.0, min(bbox.xmin, img_width))
        xmax = max(0.0, min(bbox.xmax, img_width))
        ymin = max(0.0, min(bbox.ymin, img_height))
        ymax = max(0.0, min(bbox.ymax, img_height))

        if xmax <= xmin or ymax <= ymin:
            raise ValueError("Box became invalid after clamping")

        x_center = ((xmin + xmax) / 2.0) / img_width
        y_center = ((ymin + ymax) / 2.0) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        return x_center, y_center, width, height


class YoloWriter:
    def write(self, output_path: Path, lines: List[str]) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")


class DatasetConverter:
    def __init__(
        self,
        parser: VocParser,
        converter: YoloConverter,
        writer: YoloWriter,
        skip_unknown: bool = False,
        skip_difficult: bool = False,
    ) -> None:
        self.parser = parser
        self.converter = converter
        self.writer = writer
        self.skip_unknown = skip_unknown
        self.skip_difficult = skip_difficult

    def convert_file(self, xml_path: Path, output_dir: Path) -> Tuple[int, int]:
        annotation = self.parser.parse(xml_path)

        yolo_lines: List[str] = []
        skipped = 0

        for obj in annotation.objects:
            if self.skip_difficult and obj.difficult == 1:
                skipped += 1
                continue

            try:
                line = self.converter.convert_object(
                    obj=obj,
                    img_width=annotation.width,
                    img_height=annotation.height,
                )
                yolo_lines.append(line)
            except KeyError as e:
                skipped += 1
                if not self.skip_unknown:
                    raise e

        output_file = output_dir / f"{xml_path.stem}.txt"
        self.writer.write(output_file, yolo_lines)
        return len(yolo_lines), skipped

    def convert_directory(self, annotation_dir: Path, output_dir: Path) -> None:
        # Assuming images are in a sibling folder to 'annotations'
        image_dir = annotation_dir.parent / "images" 
        
        xml_files = [f for f in annotation_dir.glob("*.xml") if not f.name.startswith('.')]
        
        if not xml_files:
            raise FileNotFoundError(f"No XML files found in {annotation_dir}")

        total_written = 0
        total_skipped = 0

        for xml_file in sorted(xml_files):
            try:
                written, skipped = self.convert_file(xml_file, output_dir)
                total_written += written
                total_skipped += skipped
            except (ET.ParseError, ValueError) as e:
                print(f"!!! Found Corrupt File: {xml_file.name}")
                total_skipped += 1
                
                # Logic to find and delete the orphaned image
                # Checks for common extensions like .jpg, .jpeg, .png
                for ext in ['.jpg', '.jpeg', '.png']:
                    img_path = image_dir / f"{xml_file.stem}{ext}"
                    if img_path.exists():
                        print(f"--- Deleting orphaned image: {img_path.name}")
                        img_path.unlink() # This deletes the file
                
                # Optional: Delete the bad XML itself so it doesn't bother you again
                # xml_file.unlink() 
                continue

        print(f"\n--- Process Complete ---")
        print(f"Labels Created: {total_written}")
        print(f"Files Skipped/Cleaned: {total_skipped}")


def main() -> None:
    annotation_dir = Path("./Lego_Project_Object_Detection/Dataset/annotations")
    output_dir = Path("./Dataset/labels")

    parser = VocParser()
    class_map_builder = ClassMapBuilder(parser)
    class_to_id = class_map_builder.build_from_directory(annotation_dir)

    print(f"Found {len(class_to_id)} classes")
    print(class_to_id)

    converter = YoloConverter(class_to_id=class_to_id)
    writer = YoloWriter()

    dataset_converter = DatasetConverter(
        parser=parser,
        converter=converter,
        writer=writer,
        skip_unknown=False,
        skip_difficult=False,
    )

    dataset_converter.convert_directory(annotation_dir, output_dir)
    print("Conversion complete")


if __name__ == "__main__":
    main()
