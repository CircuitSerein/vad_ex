# :nif tests need a built NIF (VAD_EX_BUILD=1 + a Rust toolchain, or a published precompiled
# release). :real_audio additionally needs the fetched corpus (test/corpus/fetch_corpus.sh) and
# its golden vector. Both are excluded by default:
#   mix test --include nif
#   VAD_EX_BUILD=1 mix test --include real_audio   # the local real-audio gate
ExUnit.start(exclude: [:nif, :real_audio])
