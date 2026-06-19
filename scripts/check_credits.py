#!/usr/bin/env python3
"""Check remaining credits on OpenRouter — both per-key and account-wide."""
import os
import sys
import json
import urllib.request
import urllib.error
from dotenv import load_dotenv


def fetch(url, api_key):
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode())


def main():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    # --- Per-key info ---
    try:
        key_data = fetch("https://openrouter.ai/api/v1/key", api_key)["data"]
        label = key_data.get("label", "unnamed")
        limit = key_data.get("limit")
        remaining = key_data.get("limit_remaining")
        usage = key_data.get("usage", 0)
        usage_daily = key_data.get("usage_daily", 0)
        usage_monthly = key_data.get("usage_monthly", 0)
        free_tier = key_data.get("is_free_tier", False)
        byok_usage = key_data.get("byok_usage", 0)
        byok_daily = key_data.get("byok_usage_daily", 0)
        byok_weekly = key_data.get("byok_usage_weekly", 0)
        byok_monthly = key_data.get("byok_usage_monthly", 0)

        print("=== API Key ===")
        print(f"  Key:             {label}")
        print(f"  Free tier:       {free_tier}")
        print(f"  Credit limit:    {'unlimited' if limit is None else f'${limit:.4f}'}")
        print(f"  Remaining:       {'unlimited' if remaining is None else f'${remaining:.4f}'}")
        print(f"  Total used:      ${usage:.4f}")
        print(f"  Used today:      ${usage_daily:.4f}")
        print(f"  Used this month: ${usage_monthly:.4f}")
        print()
        print("=== BYOK (Bring Your Own Key) ===")
        print(f"  BYOK total:      ${byok_usage:.4f}")
        print(f"  BYOK today:      ${byok_daily:.4f}")
        print(f"  BYOK this week:  ${byok_weekly:.4f}")
        print(f"  BYOK this month: ${byok_monthly:.4f}")
    except urllib.error.HTTPError as error:
        print(f"Key endpoint error: HTTP {error.code}: {error.read().decode()}")

    # --- Account-wide credits ---
    try:
        credits_data = fetch("https://openrouter.ai/api/v1/credits", api_key)["data"]
        total = credits_data.get("total_credits", 0)
        total_usage = credits_data.get("total_usage", 0)
        balance = total - total_usage

        print("\n=== Account ===")
        print(f"  Total credits:   ${total:.4f}")
        print(f"  Total usage:     ${total_usage:.4f}")
        print(f"  Balance:         ${balance:.4f}")
    except urllib.error.HTTPError as error:
        # /credits requires a management/provisioning key — fall back gracefully
        if error.code in (401, 403):
            print("\n=== Account ===")
            print("  (Requires a provisioning key — generate one at openrouter.ai/settings/provisioning-keys)")
        else:
            print(f"Credits endpoint error: HTTP {error.code}: {error.read().decode()}")


if __name__ == "__main__":
    main()