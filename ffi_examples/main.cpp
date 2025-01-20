// SPDX-FileCopyrightText: 2024 University of Rochester
//
// SPDX-License-Identifier: MIT

#include <iostream>
#include <stdlib.h>

#include "wgpu_abc_helper.hpp"
// using namespace wgpu_constraints;
// extern void hello_ffi(int32_t a, int32_t b);

int main()
{
    // Make a new term that is literal True
    abc_helper::Context ctx = abc_helper::abc_new_context().AsSuccess();

    abc_helper::MaybeTerm term = ctx.new_literal_true();

    if (term.IsSuccess())
    {
        auto term_inner = term.AsSuccess();
        auto term_text = ctx.term_to_cstr(term_inner);
        std::cout << "Term text: " << term_text << std::endl;
        abc_helper::abc_free_string(term_text);
    }
    else
    {
        std::cerr << "Error: " << (int)term.error._0 << std::endl;
        return EXIT_FAILURE;
    }

    // Print the term.

    return EXIT_SUCCESS;
}