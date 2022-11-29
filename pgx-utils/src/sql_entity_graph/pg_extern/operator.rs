/*
Portions Copyright 2019-2021 ZomboDB, LLC.
Portions Copyright 2021-2022 Technology Concepts & Design, Inc. <support@tcdi.com>

All rights reserved.

Use of this source code is governed by the MIT license that can be found in the LICENSE file.
*/
/*!

`#[pg_operator]` related macro expansion for Rust to SQL translation

> Like all of the [`sql_entity_graph`][crate::sql_entity_graph] APIs, this is considered **internal**
to the `pgx` framework and very subject to change between versions. While you may use this, please do it with caution.

*/
use proc_macro2::TokenStream as TokenStream2;
use quote::{quote, ToTokens, TokenStreamExt};
use syn::parenthesized;
use syn::parse::{Parse, ParseBuffer};
use syn::token::Paren;

/// A parsed `#[pg_operator]` operator.
///
/// It is created during [`PgExtern`](crate::sql_entity_graph::PgExtern) parsing.
#[derive(Debug, Default, Clone)]
pub struct PgOperator {
    pub opname: Option<PgxOperatorOpName>,
    pub commutator: Option<PgxOperatorAttributeWithIdent>,
    pub negator: Option<PgxOperatorAttributeWithIdent>,
    pub restrict: Option<PgxOperatorAttributeWithIdent>,
    pub join: Option<PgxOperatorAttributeWithIdent>,
    pub hashes: bool,
    pub merges: bool,
}

impl ToTokens for PgOperator {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let opname = self.opname.iter().clone();
        let commutator = self.commutator.iter().clone();
        let negator = self.negator.iter().clone();
        let restrict = self.restrict.iter().clone();
        let join = self.join.iter().clone();
        let hashes = self.hashes;
        let merges = self.merges;
        let quoted = quote! {
            ::pgx::utils::sql_entity_graph::PgOperatorEntity {
                opname: None #( .unwrap_or(Some(#opname.into())) )*,
                commutator: None #( .unwrap_or(Some(#commutator.into())) )*,
                negator: None #( .unwrap_or(Some(#negator.into())) )*,
                restrict: None #( .unwrap_or(Some(#restrict.into())) )*,
                join: None #( .unwrap_or(Some(#join.into())) )*,
                hashes: #hashes,
                merges: #merges,
            }
        };
        tokens.append_all(quoted);
    }
}

#[derive(Debug, Clone)]
pub struct PgxOperatorAttributeWithIdent {
    pub paren_token: Paren,
    pub fn_name: TokenStream2,
}

impl Parse for PgxOperatorAttributeWithIdent {
    fn parse(input: &ParseBuffer) -> Result<Self, syn::Error> {
        let inner;
        Ok(PgxOperatorAttributeWithIdent {
            paren_token: parenthesized!(inner in input),
            fn_name: inner.parse()?,
        })
    }
}

impl ToTokens for PgxOperatorAttributeWithIdent {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let fn_name = &self.fn_name;
        let operator = fn_name.to_string().replace(" ", "");
        let quoted = quote! {
            #operator
        };
        tokens.append_all(quoted);
    }
}

#[derive(Debug, Clone)]
pub struct PgxOperatorOpName {
    pub paren_token: Paren,
    pub op_name: TokenStream2,
}

impl Parse for PgxOperatorOpName {
    fn parse(input: &ParseBuffer) -> Result<Self, syn::Error> {
        let inner;
        Ok(PgxOperatorOpName {
            paren_token: parenthesized!(inner in input),
            op_name: inner.parse()?,
        })
    }
}

impl ToTokens for PgxOperatorOpName {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        let op_name = &self.op_name;
        let op_string = op_name.to_string().replacen(" ", "", 256);
        let quoted = quote! {
            #op_string
        };
        tokens.append_all(quoted);
    }
}
