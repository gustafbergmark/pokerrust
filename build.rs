extern crate cc;

fn main() {
    cc::Build::new()
        .cuda(true)
        .flag("-gencode")
        .flag("arch=compute_89,code=sm_89")
        .flag("--use_fast_math")
        .flag("-lineinfo")
        .flag("-ewp")
        .flag("-res-usage")
        //.flag("-dlto")
        .flag("--no-exceptions")
        //.flag("-restrict")
        .flag("--maxrregcount=80")
        .file("src/cuda/interface.cu")
        .compile("poker");

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/cuda");
}
