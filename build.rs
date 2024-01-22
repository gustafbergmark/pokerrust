extern crate cc;

fn main() {
    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode")
        .flag("arch=compute_89,code=sm_89")
        .file("src/cuda/poker.cu")
        .file("src/cuda/builder.cu")
        .compile("poker");
    println!("cargo:rerun-if-changed=src/cuda/poker.cu");
    println!("cargo:rerun-if-changed=src/cuda/builder.cu");
    println!("cargo:rerun-if-changed=src/cuda/structs.h");
}
