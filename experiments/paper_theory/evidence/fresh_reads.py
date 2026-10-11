"""A read policy for the fresh-data pipeline, including resolved symlink targets.

The Python audit hook covers Python file reads. Go scientific executables take
only generated JSON on stdin; the hook records subprocess commands, not Go
kernel-level file opens. Toolchain/dependency source reads are permitted.
"""

import os
import sys
from collections import Counter
from pathlib import Path

SOURCE_SUFFIXES = {".py", ".pyc", ".go", ".mod", ".sum", ".toml", ".pth", ".so", ".dylib"}


def within(path, root):
    return path == root or root in path.parents


class FreshReadPolicy:
    """Allow generated run files and code/dependencies; reject other data reads."""

    def __init__(self, work, publish, sources, runtimes=None, generated=()):
        self.work = Path(work).resolve()
        self.publish = Path(publish).resolve()
        self.sources = [Path(p).resolve() for p in sources]
        self.runtimes = [Path(p).resolve() for p in (runtimes or [])]
        self.generated = [Path(p).resolve() for p in generated]
        self.counts = Counter()
        self.denied = []
        self.commands = []

    def classify(self, value, directory=False):
        if isinstance(value, int):
            return "file_descriptor"
        path = Path(os.fsdecode(value)).resolve()
        if directory and not path.exists():
            return "missing_directory_probe"
        if within(path, self.work) or within(path, self.publish) or any(within(path, p) for p in self.generated):
            return "generated"
        for root in self.sources:
            if within(path, root):
                relative = path.relative_to(root)
                if directory:
                    return "source_directory"
                if relative.parts and relative.parts[0] in ("outputs", "inputs", "artifacts", "datasets", ".git"):
                    return None
                if path.suffix in SOURCE_SUFFIXES or path.name in ("go.mod", "go.sum"):
                    return "source"
                return None
        if any(within(path, root) for root in self.runtimes):
            return "runtime_dependency"
        if path in (Path("/dev/null"), Path("/dev/urandom"), Path("/dev/random")):
            return "system_device"
        if path == Path("/etc/localtime").resolve():
            return "system_timezone"
        if path == Path("/System/Library/CoreServices/SystemVersion.plist"):
            return "system_version"
        fonts = (
            Path.home() / "Library/Fonts",
            Path.home() / ".fonts",
            Path.home() / ".local/share/fonts",
            Path("/Network/Library/Fonts"),
            Path("/usr/share/fonts"),
            Path("/usr/local/share/fonts"),
            Path("/usr/X11"),
            Path("/usr/X11R6"),
            Path("/opt/local/share/fonts"),
            Path("/usr/lib/openoffice/share/fonts"),
            Path("/System/Library/PrivateFrameworks/FontServices.framework/Resources"),
        )
        fonts = tuple(root.resolve() for root in fonts)
        asset_font = (
            len(path.parts) > 4
            and path.parts[:4] == ("/", "System", "Library", "AssetsV2")
            and path.parts[4].startswith("com_apple_MobileAsset_Font")
        )
        if (asset_font or any(within(path, root) for root in fonts)) and (
            directory or path.suffix.lower() in {".ttf", ".ttc", ".otf", ".afm", ".pfb", ".pfa"}
        ):
            return "font_dependency"
        return None

    def audit(self, event, args):
        directory = event in ("os.listdir", "os.scandir")
        if event == "open":
            path, mode, flags = args
            read = ("r" in mode or "+" in mode) if isinstance(mode, str) else not (flags & os.O_WRONLY)
            if not read:
                return
        elif directory:
            path = args[0]
        elif event == "subprocess.Popen":
            self.commands.append({"executable": str(args[0]), "argv": args[1], "cwd": str(args[2])})
            return
        else:
            return
        category = self.classify(path, directory=directory)
        if category is None:
            self.denied.append(str(path))
            raise PermissionError(f"fresh-data read policy rejected: {path}")
        self.counts[category] += 1

    def install(self):
        sys.addaudithook(self.audit)

    def report(self):
        return {
            "python_read_categories": dict(self.counts),
            "denied_reads": self.denied,
            "generated_root": str(self.work),
            "temporary_generated_roots": [str(p) for p in self.generated],
            "publish_root": str(self.publish),
            "source_roots": [str(p) for p in self.sources],
            "dependency_roots": [str(p) for p in self.runtimes],
            "subprocess_commands": self.commands,
            "scope": "Python audit hook; Go scientific inputs are generated JSON via stdin",
            "symlink_policy": "classify resolved target, never trust link location",
        }


def dependency_roots():
    """Interpreter/installed packages and OS font assets, never research inputs."""
    return list(
        {
            Path(sys.prefix).resolve(),
            Path(sys.base_prefix).resolve(),
            Path("/System/Library/Fonts"),
            Path("/Library/Fonts"),
            Path("/System/Library/Frameworks"),
            Path("/usr/share/zoneinfo"),
        }
    )
