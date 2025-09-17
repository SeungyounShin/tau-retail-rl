import argparse
import os
import re
import sys
from typing import List, Tuple, Optional

try:
	import litellm
except Exception as e:  # pragma: no cover
	litellm = None

# Optional progress bar
try:
	from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
	tqdm = None

# Optional colored output
try:
	from colorama import Fore, Style, init as colorama_init  # type: ignore
	_COLORAMA_AVAILABLE = True
except Exception:  # pragma: no cover
	_COLORAMA_AVAILABLE = False
	class _Dummy:  # minimal fallback
		RESET_ALL = ""
		BLACK = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = ""
	Fore = _Dummy()
	Style = _Dummy()
	def colorama_init(*args, **kwargs):
		return None


TASK_BLOCK_START_RE = re.compile(r"^(\s*)Task\(\s*$")
INSTRUCTION_LINE_RE = re.compile(r"^(\s*)instruction\s*=\s*(?P<quote>['\"])", re.DOTALL)
PROMPT_LINE_RE = re.compile(r"^(\s*)prompt\s*=\s*(?P<quote>['\"])", re.DOTALL)


def _colorize(text: str, color: str, enable: bool) -> str:
	return f"{color}{text}{Style.RESET_ALL}" if enable else text


def _truncate(text: str, max_len: int) -> str:
	if max_len is None or max_len <= 0:
		return text
	return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _find_task_blocks(text: str) -> List[Tuple[int, int]]:
	"""Return list of (start_idx, end_idx_exclusive) slices for each Task(...) block.
	Uses simple parenthesis counting starting at occurrences of 'Task(' at line starts.
	"""
	blocks: List[Tuple[int, int]] = []
	i = 0
	n = len(text)
	while True:
		start = text.find("Task(", i)
		if start == -1:
			break
		# ensure 'Task(' is at line start or preceded by whitespace/comma
		line_start = text.rfind("\n", 0, start) + 1
		# Count parentheses until balanced
		depth = 0
		j = start
		while j < n:
			ch = text[j]
			if ch == '"' or ch == "'":
				# skip string literal
				quote = ch
				j += 1
				while j < n:
					if text[j] == "\\":
						j += 2
						continue
					if text[j] == quote:
						j += 1
						break
					j += 1
				continue
			if ch == '(':
				depth += 1
			elif ch == ')':
				depth -= 1
				if depth == 0:
					# include trailing comma if present
					end = j + 1
					if end < n and text[end] == ',':
						end += 1
					# include following whitespace/newlines
					while end < n and text[end] in " \t\r\n":
						end += 1
					blocks.append((line_start, end))
					i = end
					break
			j += 1
		else:
			# Unterminated block, stop
			break
	return blocks


def _extract_line_value(line: str) -> Optional[str]:
	"""Extract string literal value from a line like: instruction="..." or prompt='...'
	Returns the unescaped inner string, or None if not a simple one-line literal.
	"""
	m = re.search(r"=['\"](.*)['\"]\s*,?\s*$", line)
	if not m:
		return None
	raw = m.group(1)
	# Unescape basic sequences
	return bytes(raw, "utf-8").decode("unicode_escape")


def _escape_string(value: str) -> str:
	return (
		value.replace("\\", "\\\\")
		.replace("\n", "\\n")
		.replace("\r", "\\r")
		.replace("\t", "\\t")
		.replace('"', '\\"')
	)


def _generate_prompt(instruction: str, model: str, temperature: float, max_tokens: int) -> str:
	if litellm is None:
		raise RuntimeError(
			"litellm is not installed. Please `pip install litellm` and set your API key(s)."
		)

	system_prompt = (
		"You are a data augmentation assistant. Rewrite the given instruction into a concise, natural, first-person user message to a retail customer service agent. "
		"Do not include meta commentary or step-by-step plans. Keep product/order IDs and factual details as-is. "
		"You must tell your identity to the customer service agent. (name, zip, email)"
        "Aim for 1-2 sentences max."
	)

	try:
		resp = litellm.completion(
			model=model,
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": instruction},
			],
			temperature=temperature,
			max_tokens=max_tokens,
		)
	except Exception as e:
		raise RuntimeError(f"litellm.completion failed: {e}")

	# Support both chat and text completions shape
	content = None
	try:
		content = resp["choices"][0]["message"]["content"]
	except Exception:
		try:
			content = resp["choices"][0]["text"]
		except Exception:
			pass
	if not content:
		raise RuntimeError("Could not extract content from litellm response")

	# Normalize to single line
	content = content.strip()
	content = re.sub(r"\s+", " ", content)
	return content


def _extract_identity_from_block(block_text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
	"""Parse first_name, last_name, zip from a Task block's actions if present."""
	first = None
	last = None
	zip_code = None
	m = re.search(r'"first_name"\s*:\s*"([^"]+)"', block_text)
	if m:
		first = m.group(1)
	m = re.search(r'"last_name"\s*:\s*"([^"]+)"', block_text)
	if m:
		last = m.group(1)
	m = re.search(r'"zip"\s*:\s*"(\d{5})"', block_text)
	if m:
		zip_code = m.group(1)
	return first, last, zip_code


def _prepend_identity_if_missing(text: str, first: Optional[str], last: Optional[str], zip_code: Optional[str]) -> str:
	if not first and not last and not zip_code:
		return text
	need_name = True
	need_zip = True
	full_name = None
	if first or last:
		full_name = (first or "").strip() + (" " if (first and last) else "") + (last or "").strip()
		if full_name and full_name.lower() in text.lower():
			need_name = False
	if zip_code and (zip_code in text):
		need_zip = False
	if not need_name and not need_zip:
		return text
	parts = []
	if need_name and full_name:
		parts.append(f"Hi my name is {full_name}")
	if need_zip and zip_code:
		if parts:
			parts.append(f"and my zip code is {zip_code}")
		else:
			parts.append(f"My zip code is {zip_code}")
	prefix = ", ".join(parts)
	prefix = (prefix + ". ") if prefix else ""
	return (prefix + text) if prefix else text


def _process_task_block(block_text: str, model: str, temperature: float, max_tokens: int, force: bool) -> Tuple[str, bool, Optional[str], Optional[str]]:
	"""Process one Task(...) block text. Returns (new_block_text, changed?, instruction, generated)."""
	lines = block_text.splitlines()

	instruction_idx = None
	prompt_idx = None

	for idx, line in enumerate(lines):
		if instruction_idx is None and line.strip().startswith("instruction="):
			instruction_idx = idx
		if prompt_idx is None and line.strip().startswith("prompt="):
			prompt_idx = idx
		if instruction_idx is not None and prompt_idx is not None:
			break

	if instruction_idx is None:
		return block_text, False, None, None

	instruction_line = lines[instruction_idx]
	instruction_value = _extract_line_value(instruction_line)
	if instruction_value is None:
		return block_text, False, None, None

	if prompt_idx is not None and not force:
		return block_text, False, instruction_value, None

	generated = _generate_prompt(
		instruction=instruction_value, model=model, temperature=temperature, max_tokens=max_tokens
	)

	# Enforce identity (name + zip) from actions if present
	first, last, zip_code = _extract_identity_from_block(block_text)
	generated = _prepend_identity_if_missing(generated, first, last, zip_code)

	# Compose prompt line using the same indentation as instruction line
	indent = instruction_line[: len(instruction_line) - len(instruction_line.lstrip(" \t"))]
	prompt_line = f'{indent}prompt="{_escape_string(generated)}",'

	if prompt_idx is None:
		# Insert after instruction line
		lines.insert(instruction_idx + 1, prompt_line)
	else:
		lines[prompt_idx] = prompt_line

	return "\n".join(lines), True, instruction_value, generated


def process_file(path: str, model: str, force: bool, dry_run: bool, temperature: float, max_tokens: int, limit: Optional[int], verbose: bool, use_color: bool, show_progress: bool, truncate_len: int) -> int:
	with open(path, "r", encoding="utf-8") as f:
		original = f.read()

	blocks = _find_task_blocks(original)
	if not blocks:
		print(f"No Task(...) blocks found in {path}")
		return 0

	changed_total = 0
	new_parts: List[str] = []
	cursor = 0
	processed_count = 0

	iterator = enumerate(blocks)
	if show_progress and tqdm is not None:
		iterator = enumerate(tqdm(blocks, desc=os.path.basename(path), leave=False))

	for idx, (start, end) in iterator:  # type: ignore
		# Append content before block unchanged
		new_parts.append(original[cursor:start])
		block_text = original[start:end]

		# Respect limit
		do_process = True
		if limit is not None and processed_count >= limit:
			do_process = False

		if do_process:
			try:
				new_block, changed, instr_val, gen_val = _process_task_block(
					block_text, model=model, temperature=temperature, max_tokens=max_tokens, force=force
				)
			except Exception as e:
				print(f"[WARN] Failed to process a Task block in {path}: {e}")
				new_block, changed, instr_val, gen_val = block_text, False, None, None
		else:
			new_block, changed, instr_val, gen_val = block_text, False, None, None

		if changed:
			changed_total += 1
			processed_count += 1
			if verbose:
				status = "WRITE" if not dry_run else "DRY-RUN"
				status_col = Fore.GREEN if not dry_run else Fore.YELLOW
				file_tag = _colorize(f"[{os.path.basename(path)}]", Fore.CYAN, use_color)
				print(f"{file_tag} Task {idx + 1}: " + _colorize(status, status_col, use_color))
				if instr_val is not None:
					print("  " + _colorize("instruction", Fore.MAGENTA, use_color) + ": " + _colorize(_truncate(instr_val, truncate_len), Fore.YELLOW, use_color))
				if gen_val is not None:
					print("  " + _colorize("prompt", Fore.MAGENTA, use_color) + ": " + _colorize(_truncate(gen_val, truncate_len), Fore.GREEN, use_color))

		new_parts.append(new_block)
		cursor = end

	# Remainder
	new_parts.append(original[cursor:])
	new_text = "".join(new_parts)

	if changed_total > 0 and not dry_run:
		backup_path = path + ".bak"
		with open(backup_path, "w", encoding="utf-8") as f:
			f.write(original)
		with open(path, "w", encoding="utf-8") as f:
			f.write(new_text)
		print(f"Updated {path} (changed {changed_total} Task blocks). Backup at {backup_path}")
	else:
		print(f"Scanned {path} (changed {changed_total} Task blocks).{' [dry-run]' if dry_run else ''}")

	return changed_total


def main(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(
		description="Generate or update prompt= from instruction= in Task(...) blocks using litellm."
	)
	parser.add_argument(
		"--files",
		nargs="+",
		default=[
			os.path.join(os.path.dirname(__file__), "tasks_test.py"),
			os.path.join(os.path.dirname(__file__), "tasks_train.py"),
		],
		help="Paths to Python files containing Task(...) definitions.",
	)
	parser.add_argument("--model", default=os.environ.get("LITELLM_MODEL", "gpt-3.5-turbo"))
	parser.add_argument("--force", action="store_true", help="Regenerate even if prompt= already exists.")
	parser.add_argument("--dry-run", action="store_true", help="Do not write files; just report changes.")
	parser.add_argument("--temperature", type=float, default=0.2)
	parser.add_argument("--max-tokens", type=int, default=120)
	parser.add_argument("--limit", type=int, default=None, help="Max number of Task blocks to modify per file.")
	parser.add_argument("--verbose", action="store_true", help="Print instruction→prompt mapping for each changed block.")
	parser.add_argument("--no-color", action="store_true", help="Disable ANSI colored output.")
	parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar.")
	parser.add_argument("--truncate", type=int, default=220, help="Max characters to show per text field in logs.")

	args = parser.parse_args(argv)

	# init color if available and enabled
	use_color = (not args.no_color) and _COLORAMA_AVAILABLE and sys.stdout.isatty()
	if use_color:
		try:
			colorama_init(autoreset=True)
		except Exception:
			use_color = False

	show_progress = (not args.no_progress) and (tqdm is not None) and sys.stderr.isatty()

	total_changed = 0

	for path in args.files:
		if not os.path.isabs(path):
			path = os.path.abspath(path)
		if not os.path.exists(path):
			print(f"[WARN] File not found: {path}")
			continue
		total_changed += process_file(
			path=path,
			model=args.model,
			force=args.force,
			dry_run=args.dry_run,
			temperature=args.temperature,
			max_tokens=args.max_tokens,
			limit=args.limit,
			verbose=args.verbose,
			use_color=use_color,
			show_progress=show_progress,
			truncate_len=args.truncate,
		)

	return 0 if total_changed >= 0 else 1


if __name__ == "__main__":  # pragma: no cover
	sys.exit(main()) 