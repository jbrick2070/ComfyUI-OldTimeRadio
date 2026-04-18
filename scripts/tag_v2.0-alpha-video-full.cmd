@echo off
REM ============================================================================
REM Day 14 release tag handoff -- RUN THIS YOURSELF, JEFFREY.
REM Per CLAUDE.md: "Only Jeffrey merges to main and tags releases."
REM ============================================================================
REM
REM What this does:
REM   1. Verify working tree clean on v2.0-alpha-video-stack.
REM   2. Verify local HEAD == origin/v2.0-alpha-video-stack (lockstep).
REM   3. Annotated-tag the current HEAD as v2.0-alpha-video-full.
REM   4. Push the tag to origin.
REM
REM If any step fails, the script stops -- no partial-tag state on origin.
REM ============================================================================

cd /d C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
if errorlevel 1 (
    echo [FAIL] Could not cd into the repo.
    exit /b 1
)

echo.
echo === 1/4: Checking current branch ===
for /f "tokens=*" %%b in ('git rev-parse --abbrev-ref HEAD') do set CUR_BRANCH=%%b
if not "%CUR_BRANCH%"=="v2.0-alpha-video-stack" (
    echo [FAIL] Expected branch v2.0-alpha-video-stack, got %CUR_BRANCH%.
    echo        Run: git checkout v2.0-alpha-video-stack
    exit /b 1
)
echo [OK] On v2.0-alpha-video-stack.

echo.
echo === 2/4: Checking working tree is clean ===
git diff --quiet
if errorlevel 1 (
    echo [FAIL] Working tree has unstaged changes. Commit or stash first.
    exit /b 1
)
git diff --cached --quiet
if errorlevel 1 (
    echo [FAIL] Working tree has staged-but-uncommitted changes. Commit first.
    exit /b 1
)
echo [OK] Working tree clean.

echo.
echo === 3/4: Verifying lockstep with origin ===
git fetch origin v2.0-alpha-video-stack
for /f "tokens=*" %%h in ('git rev-parse HEAD') do set LOCAL_HEAD=%%h
for /f "tokens=*" %%h in ('git rev-parse origin/v2.0-alpha-video-stack') do set ORIGIN_HEAD=%%h
if not "%LOCAL_HEAD%"=="%ORIGIN_HEAD%" (
    echo [FAIL] Local HEAD %LOCAL_HEAD% != origin HEAD %ORIGIN_HEAD%.
    echo        Push local commits or pull origin before tagging.
    exit /b 1
)
echo [OK] Lockstep: %LOCAL_HEAD%

echo.
echo === 4/4: Tagging v2.0-alpha-video-full and pushing ===
git tag -a v2.0-alpha-video-full -m "v2.0-alpha video stack feature-complete (Days 1-14, 2026-04-17)"
if errorlevel 1 (
    echo [FAIL] Tag creation failed. Tag may already exist locally.
    echo        To retag: git tag -d v2.0-alpha-video-full
    exit /b 1
)
git push origin v2.0-alpha-video-full
if errorlevel 1 (
    echo [FAIL] Tag push failed. Tag may already exist on origin.
    exit /b 1
)

echo.
echo === DONE ===
echo Tagged %LOCAL_HEAD% as v2.0-alpha-video-full on origin.
echo Verify: https://github.com/jbrick2070/ComfyUI-OldTimeRadio/releases/tag/v2.0-alpha-video-full
