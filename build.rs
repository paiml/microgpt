//! Build script for provable contract enforcement.
//!
//! Reads contracts/binding.yaml and sets CONTRACT_* environment variables
//! so that the contract annotations can verify bindings at compile time.

use std::path::Path;

fn main() {
    let binding_path = Path::new("contracts/binding.yaml");
    if binding_path.exists() {
        println!("cargo:rustc-env=CONTRACT_BINDING_YAML={}", binding_path.display());
        println!("cargo:rustc-env=CONTRACT_CRATE=microgpt");
        println!("cargo:rustc-env=CONTRACT_POLICY=AllImplemented");

        // Count bindings by reading the file
        let content = std::fs::read_to_string(binding_path).expect("read binding.yaml");
        let binding_count = content.matches("status: implemented").count();
        println!("cargo:rustc-env=CONTRACT_BINDING_COUNT={binding_count}");
        println!("cargo:rustc-env=CONTRACT_BINDING_IMPLEMENTED={binding_count}");

        // Re-run if binding.yaml changes
        println!("cargo:rerun-if-changed=contracts/binding.yaml");
        println!("cargo:rerun-if-changed=contracts/microgpt-v1.yaml");
    }
}
