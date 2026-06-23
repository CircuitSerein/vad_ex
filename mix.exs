defmodule VadEx.MixProject do
  use Mix.Project

  @version "0.1.0-dev"
  @source_url "https://github.com/CircuitSerein/vad_ex"

  def project do
    [
      app: :vad_ex,
      version: @version,
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description:
        "Streaming Voice Activity Detection + endpointing for the BEAM. " <>
          "Silero VAD via a Rustler/ONNX-Runtime NIF, with a GenServer-per-stream " <>
          "API and a Membrane filter.",
      package: package(),
      name: "VadEx",
      source_url: @source_url,
      docs: docs()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      # NIF distribution: ship precompiled binaries, no Rust toolchain needed by users.
      {:rustler_precompiled, "~> 0.9"},
      # Only needed to BUILD the NIF locally (VAD_EX_BUILD=1). Optional for consumers.
      # Floor bumped to 0.38 to match the Rust crate (Resource trait + #[resource_impl] API).
      {:rustler, ">= 0.38.0", optional: true},
      # Telemetry events ([:vad_ex, :chunk | :speech_start | :speech_end]).
      {:telemetry, "~> 1.2"},
      # Membrane filter is an OPTIONAL integration — core works without it.
      {:membrane_core, "~> 1.3", optional: true},
      {:membrane_raw_audio_format, "~> 0.12", optional: true},
      # Docs/dev only.
      {:ex_doc, "~> 0.34", only: :dev, runtime: false}
    ]
  end

  defp package do
    [
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url},
      # Ship sources + the bundled Silero model + the checksum guard. NOT target/.
      files: ~w(lib native priv/models .formatter.exs mix.exs README.md LICENSE
                CHANGELOG.md checksum-Elixir.VadEx.Native.exs)
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: ["README.md", "CHANGELOG.md", "docs/architecture.md"],
      groups_for_modules: [
        Core: [VadEx, VadEx.Session, VadEx.Endpointer],
        Membrane: [VadEx.Membrane.Filter],
        Internals: [VadEx.Native]
      ]
    ]
  end
end
