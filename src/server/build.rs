fn main() -> Result<(), Box<dyn std::error::Error>> {
    // When publishing the crate, files outside the crate root (e.g. top-level /proto)
    // are not included in the package. Copy the proto files into the crate at
    // `proto/` and refer to them relative to the crate manifest so packaging
    // verification succeeds.
    let proto_file = "proto/vectradb.proto";
    let proto_dir = "proto";

    // Use OUT_DIR for generated files (standard practice)
    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .compile_protos(&[proto_file], &[proto_dir])?;

    println!("cargo:rerun-if-changed={}", proto_file);
    Ok(())
}
