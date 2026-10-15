@testitem "Examples run without errors" begin
    # SGM currently requires LinearAlgebra 1.12; keep the package's Julia 1.10 tests.
    if VERSION < v"1.12"
        @test_skip VERSION >= v"1.12"
    else
        root = pkgdir(IPG)
        examples_dir = joinpath(root, "examples")
        # Include new tracked examples automatically, excluding untracked local scripts.
        paths = split(read(`git -C $root ls-files -z -- examples/`, String), '\0'; keepempty = false)
        examples = sort(filter(path -> dirname(path) == "examples" && endswith(path, ".jl"), paths))
        @test !isempty(examples)

        timeout = parse(Float64, get(ENV, "IPG_EXAMPLE_TIMEOUT", "600"))
        isfinite(timeout) && timeout > 0 || error("IPG_EXAMPLE_TIMEOUT must be finite and positive")
        log_dir = get(() -> mktempdir(; cleanup = false), ENV, "IPG_EXAMPLE_LOG_DIR")
        mkpath(log_dir)
        println("Example logs: $log_dir")

        for example in examples
            @testset "$example" begin
                path = joinpath(root, example)
                log_path = joinpath(log_dir, basename(example) * ".log")
                # Use normal script settings instead of inheriting Pkg.test's compiler flags.
                julia = first(Base.julia_cmd())
                command = Cmd(
                    `$julia --startup-file=no --project=$examples_dir $path`;
                    dir = examples_dir,
                )
                println("Running $example (timeout: $timeout seconds)")
                passed, timed_out = open(log_path, "w") do log
                    process = run(pipeline(ignorestatus(command); stdout = log, stderr = log); wait = false)
                    timed_out = timedwait(() -> process_exited(process), timeout; pollint = 0.1) == :timed_out
                    timed_out && kill(process, Base.SIGKILL)
                    wait(process)
                    return success(process), timed_out
                end
                if !passed || timed_out
                    println(stderr, "Failed $example (timeout: $timed_out):")
                    println(stderr, read(log_path, String))
                end
                @test !timed_out
                @test passed
            end
        end
    end
end
