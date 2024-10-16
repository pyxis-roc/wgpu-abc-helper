// SPDX-FileCopyrightText: 2024 University of Rochester
//
// SPDX-License-Identifier: MIT

#include <iostream>

#include "wgpu_constraints.hpp"
// using namespace wgpu_constraints;
// extern void hello_ffi(int32_t a, int32_t b);

int main() {
    // Make a new term that is literal True
    abc_helper::MaybeTerm term = abc_helper::new_literal_true();
    if (term.tag == abc_helper::MaybeTerm::Tag::Term) {
        std::cout << "Term ID: " << term.term._0.id << std::endl;
        auto term_inner = term.term._0;
        auto term_text = abc_helper::to_c_str(term_inner);
        std::cout << "Term text: " << term_text << std::endl;
        abc_helper::free_string(term_text);

    } else {
        std::cout << "Error: " << (int)term.error._0 << std::endl;
    }



    // Print the term.

    
    return 0;
}