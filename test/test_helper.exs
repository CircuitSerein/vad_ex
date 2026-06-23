# :nif tests need a built NIF (VAD_EX_BUILD=1 + a Rust toolchain + ORT_DYLIB_PATH, or a published
# precompiled release). They are excluded by default; run them with `mix test --include nif`.
ExUnit.start(exclude: [:nif])
