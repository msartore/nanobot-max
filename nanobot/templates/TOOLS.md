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

## cron — Scheduled Reminders

- Please refer to cron skill for usage.
