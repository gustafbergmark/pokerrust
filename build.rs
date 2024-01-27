extern crate cc;

fn main() {
    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode")
        .flag("arch=compute_89,code=sm_89")
        .flag("-rdc=true")
        .flag("--use_fast_math")
        .flag("-lineinfo")
        .flag("-maxrregcount=40")
        .file("src/cuda/poker.cu")
        .file("src/cuda/builder.cu")
        .file("src/cuda/evaluator.cu")
        .compile("poker");

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cuda");
}
