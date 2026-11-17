.PHONY: help doctor test

help: ## List the documented targets
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

doctor: ## Check that this machine can run the skill and that Claude Code can find it (read-only)
	./doctor.sh

test: ## Run the unit tests, then check doctor.sh (syntax, and one run against an empty HOME)
	python3 -m unittest discover -s scripts/tests
	bash -n doctor.sh
	@tmp=$$(mktemp -d) && { HOME="$$tmp" ./doctor.sh > /dev/null; status=$$?; rm -rf "$$tmp"; exit $$status; }
