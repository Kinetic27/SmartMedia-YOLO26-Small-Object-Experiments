from __future__ import annotations

import py_compile
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = [
    'STA' + 'GING', 'stag' + 'ing', 'MANI' + 'FEST', 'EXCLUDED' + '_ASSETS', 'PROV' + 'ENANCE',
    '/home/' + 'prml', 'Seok' + 'Jin', '<LOC' + 'AL', 'LOCAL' + '_WORKSPACE', 'O' + 'MX',
    'dis' + 'cord', 'web' + 'hook', 'TO' + 'KEN', 'PASS' + 'WORD', 'SEC' + 'RET',
    'PRIVATE' + ' KEY', 'Next' + 'Cloud', 'Web' + 'DAV', 'Label' + ' Studio',
    'label' + ' studio', 'AI' + 'Hub', '.' + 'omx', 'yolo' + '11', 'YOLO' + '11',
]


class TestPublicContract(unittest.TestCase):
    def test_no_private_markers(self) -> None:
        for path in ROOT.rglob('*'):
            if path.is_dir() or '.git' in path.parts or '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.pt', '.pth'}:
                self.fail(f'unexpected artifact tracked: {path.relative_to(ROOT)}')
            text = path.read_text(encoding='utf-8', errors='ignore')
            for marker in FORBIDDEN:
                self.assertNotIn(marker, text, f'forbidden marker found in {path.relative_to(ROOT)}')

    def test_expected_public_files(self) -> None:
        expected = [
            'README.md', 'RESULTS.md', 'LICENSE', 'requirements.txt',
            'configs/visdrone.yaml', 'configs/yolo26.yaml',
            'configs/yolo26-p2.yaml', 'configs/yolo26-spd.yaml',
            'configs/yolo26-spd-p2.yaml', 'scripts/run_train.py',
            'scripts/prepare_visdrone.py', 'scripts/analyze_small_objects.py',
            'scripts/run_official_yolo26n_recipe_approx.sh',
        ]
        for rel in expected:
            self.assertTrue((ROOT / rel).exists(), rel)
        self.assertFalse((ROOT / 'configs/yolo26_spd_p2_nop5_eca_p2.yaml').exists())

    def test_python_scripts_compile(self) -> None:
        for script in (ROOT / 'scripts').glob('*.py'):
            py_compile.compile(str(script), doraise=True)

    def test_shell_scripts_parse(self) -> None:
        for script in (ROOT / 'scripts').glob('*.sh'):
            subprocess.run(['bash', '-n', str(script)], check=True)

    def test_analyze_small_objects_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp) / 'labels'
            labels.mkdir()
            (labels / 'sample.txt').write_text('0 0.5 0.5 0.01 0.01\n0 0.5 0.5 0.5 0.5\n')
            proc = subprocess.run(
                ['python3', str(ROOT / 'scripts/analyze_small_objects.py'), '--labels-dir', str(labels)],
                check=True,
                text=True,
                capture_output=True,
            )
            self.assertIn('very_small', proc.stdout)
            self.assertIn('large', proc.stdout)


if __name__ == '__main__':
    unittest.main()
