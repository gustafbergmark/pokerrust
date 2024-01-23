extern crate cc;

fn main() {
    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode")
        .flag("arch=compute_89,code=sm_89")
        .file("src/cuda/poker.cu")
        .file("src/cuda/builder.cu")
        .file("src/cuda/evaluator.cu")
        .compile("poker");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cuda/poker.cu");
    println!("cargo:rerun-if-changed=src/cuda/builder.cu");
    println!("cargo:rerun-if-changed=src/cuda/structs.h");
    println!("cargo:rerun-if-changed=src/cuda/evaluator.cu");
    println!("cargo:rerun-if-changed=src/cuda/evaluator.cuh");
}
