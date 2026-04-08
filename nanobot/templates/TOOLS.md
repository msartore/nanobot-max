# Tool Usage Notes

Tool signatures are provided automatically via function calling.
This file documents non-obvious constraints and usage patterns.

## message — Sending Files to the User

When the user asks to receive a file (e.g. "send me the report", "download that file", "share the CSV"):

- Use the `message` tool with the `media` parameter containing the file path(s)
- Both absolute paths and workspace-relative paths are accepted (e.g. `memory/finance/report.md`)
- The file will be delivered as a document attachment on the user's channel (Telegram, Discord, etc.)
- Include a brief text in `content` describing what you are sending

Example:
```
message(content="Here is the market intelligence log.", media=["memory/finance/market-intelligence-log.md"])
```

Do NOT use `read_file` to send files — that only reads content for your own analysis and pastes it as text.

## exec — Safety Limits

- Commands have a configurable timeout (default 60s)
- Dangerous commands are blocked (rm -rf, format, dd, shutdown, etc.)
- Output is truncated at 10,000 characters
- `restrictToWorkspace` config can limit file access to the workspace

## glob — File Discovery

- Use `glob` to find files by pattern before falling back to shell commands
- Simple patterns like `*.py` match recursively by filename
- Use `entry_type="dirs"` when you need matching directories instead of files
- Use `head_limit` and `offset` to page through large result sets
- Prefer this over `exec` when you only need file paths

## grep — Content Search

- Use `grep` to search file contents inside the workspace
- Default behavior returns only matching file paths (`output_mode="files_with_matches"`)
- Supports optional `glob` filtering plus `context_before` / `context_after`
- Supports `type="py"`, `type="ts"`, `type="md"` and similar shorthand filters
- Use `fixed_strings=true` for literal keywords containing regex characters
- Use `output_mode="files_with_matches"` to get only matching file paths
- Use `output_mode="count"` to size a search before reading full matches
- Use `head_limit` and `offset` to page across results
- Prefer this over `exec` for code and history searches
- Binary or oversized files may be skipped to keep results readable

## web_fetch vs htmlunit_fetch — Choosing the Right Web Tool

`web_fetch` is the default tool for fetching web pages. Use it first.

Use `htmlunit_fetch` when:
- `web_fetch` returns empty, minimal, or clearly incomplete content
- The page is known to be a Single-Page Application (SPA) or heavily JavaScript-rendered (React, Vue, Angular, etc.)
- The page shows a loading spinner / "JavaScript required" message when fetched without JS
- You need to scrape content that is only rendered after JS execution (e.g. dynamic tables, lazy-loaded text)

`htmlunit_fetch` runs a real headless browser (HtmlUnit/Java) that executes JavaScript before returning the page.
It is slower than `web_fetch` (~5–30 s) so only use it when `web_fetch` is insufficient.

Parameters:
- `url` — required
- `jsWaitMs` — milliseconds to wait for JS to finish (default 3 000; increase to 8 000–15 000 for slow SPAs)
- `maxChars` — max characters of output (default 50 000)

## cron — Scheduled Reminders

- Please refer to cron skill for usage.
