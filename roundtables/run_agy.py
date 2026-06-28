import sys
import asyncio
import argparse
from google.antigravity import Agent, LocalAgentConfig, CapabilitiesConfig

async def main():
    parser = argparse.ArgumentParser(description="Headless wrapper for Antigravity Agent")
    parser.add_argument("--prompt", type=str, help="Prompt text")
    parser.add_argument("--prompt-file", type=str, help="File containing the prompt text")
    args = parser.parse_args()

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read()
    elif args.prompt:
        prompt = args.prompt
    else:
        print("Error: --prompt or --prompt-file required", file=sys.stderr)
        sys.exit(1)

    # Use default capabilities (tools) and base instructions
    config = LocalAgentConfig(
        system_instructions="You are a roundtable panelist and expert coding assistant. Follow instructions precisely.",
        capabilities=CapabilitiesConfig()
    )

    try:
        async with Agent(config) as agent:
            response = await agent.chat(prompt)
            async for chunk in response:
                sys.stdout.write(chunk)
            print()
    except Exception as e:
        print(f"Agent error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
