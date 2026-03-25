.PHONY: bench-readme

bench-readme: ## Update README.md with benchmark results from results/*.csv
	@TABLE=$$(uv run python3 scripts/gen-benchmark-table.py) && \
	uv run python3 -c "\
import sys, re; \
readme = open('README.md').read(); \
table = sys.argv[1]; \
updated = re.sub( \
    r'(<!-- BENCH:START -->).*?(<!-- BENCH:END -->)', \
    r'\1\n' + table + r'\n\2', \
    readme, flags=re.DOTALL); \
open('README.md', 'w').write(updated)" "$$TABLE"
	@echo "README.md updated with benchmark table"