
extern crate parse_input;

use parse_input::{Input, ParseError, PSym};

#[derive(Debug)]
pub struct RowItem {
    pub slot : PSym,
    pub slot_type : Type, 
}

#[derive(Debug)]
pub enum Type {
    Unit,
    Simple(PSym),
    Index(PSym, Vec<Type>),
    Fun { input : Vec<Type>, output : Box<Type> },
    Tuple(Vec<Type>), 
    Namespace(Vec<PSym>, Box<Type>),
    Row(Vec<RowItem>),
    Infer,
}


fn parse_tuple_type(input : &mut Input) -> Result<Type, ParseError> {
    input.expect("(")?;
    let mut types = input.list(parse_type)?;
    input.expect(")")?;
    
    if types.len() == 0 {
        Ok(Type::Unit)
    }
    else if types.len() == 1 {
        Ok(types.pop().unwrap())
    }
    else {
        Ok(Type::Tuple(types))
    }
}

fn parse_fun_type(parse_input: &mut Input) -> Result<Type, ParseError> {
    parse_input.expect("fun")?;
    parse_input.expect("(")?;
    let input = parse_input.list(parse_type)?;
    parse_input.expect(")")?;
    parse_input.expect("->")?;
    let output = parse_type(parse_input)?;

    Ok(Type::Fun { input, output: Box::new(output) })
}

fn parse_namespace_type(input : &mut Input, init : PSym) -> Result<(Vec<PSym>, PSym), ParseError> {
    let mut names = vec![];

    names.push( init );

    loop {
        let sym = input.parse_symbol()?;
        match input.expect("::") {
            Ok(_) => names.push( sym ),
            Err(_) => return Ok((names, sym)),
        }
    }
}

fn parse_index_type(input : &mut Input, init : PSym) -> Result<Type, ParseError> {

    let indices = input.list(parse_type)?;

    input.expect(">")?;

    Ok(Type::Index( init, indices ))
}

pub fn parse_type(input : &mut Input) -> Result<Type, ParseError> {

    match parse_tuple_type(input) {
        Ok(t) => return Ok(t),
        _ => (),
    }

    match parse_fun_type(input) {
        Ok(t) => return Ok(t),
        _ => (),
    }

    let simple = input.parse_symbol()?;

    match input.expect("::") {
        Ok(_) => {
            let (namespace, symbol) = parse_namespace_type(input, simple)?;
            match input.expect("<") {
                Ok(_) => {
                    let index_type = parse_index_type(input, symbol)?;
                    Ok(Type::Namespace(namespace, Box::new(index_type)))
                },
                Err(_) => Ok(Type::Namespace(namespace, Box::new(Type::Simple(symbol)))),
            }
        },
        Err(_) =>
            match input.expect("<") {
                Ok(_) => Ok(parse_index_type(input, simple)?),
                Err(_) => Ok(Type::Simple(simple)),
            },
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn sym_proj( v : &PSym ) -> String {
        match v {
            PSym { value, .. } => value.to_string(),
        }
    }

    #[test]
    fn should_parse_simple_type() -> Result<(), ParseError> {
        let i = "simple".char_indices().collect::<Vec<(usize, char)>>();
        let mut input = Input::new(&i);
        let u = parse_type(&mut input)?;
        let name = match u {
            Type::Simple(PSym { value, .. }) => value,
            _ => panic!("should be simple type"), 
        };
        assert_eq!( name, "simple" );
        Ok(())
    }

    #[test]
    fn should_parse_indexed_type() -> Result<(), ParseError> {
        let i = "simple<alpha, beta>".char_indices().collect::<Vec<(usize, char)>>();
        let mut input = Input::new(&i);
        let u = parse_type(&mut input)?;
        let (name, types) = match u {
            Type::Index(PSym { value, .. }, ts) => (value, ts),
            _ => panic!("should be indexed type"),
        };
        assert_eq!( name, "simple" );
        assert_eq!( types.len(), 2 );

        let i0_name = match &types[0] {
            Type::Simple(PSym { value, .. }) => value,
            _ => panic!("index 0 should be simple type"),
        };
        
        let i1_name = match &types[1] {
            Type::Simple(PSym { value, .. }) => value,
            _ => panic!("index 1 should be simple type"),
        };

        assert_eq!( i0_name, "alpha" );
        assert_eq!( i1_name, "beta" );

        Ok(())
    }

    #[test]
    fn should_parse_namespace_type() -> Result<(), ParseError> {
        let i = "mod1::mod2::Trait::Type".char_indices().collect::<Vec<(usize, char)>>();
        let mut input = Input::new(&i);
        let u = parse_type(&mut input)?;
        let (names, t) = match u {
            Type::Namespace(ns, t) => (ns, t),
            _ => panic!("should be namespace type"),
        };

        assert_eq!( names.len(), 3 );
        assert_eq!( sym_proj(&names[0]), "mod1" );
        assert_eq!( sym_proj(&names[1]), "mod2" );
        assert_eq!( sym_proj(&names[2]), "Trait" );

        let st_name = match *t {
            Type::Simple(PSym { value, .. }) => value,
            _ => panic!("type should be simple type"),
        };

        assert_eq!( st_name, "Type" );

        Ok(())
    }

    #[test]
    fn should_parse_unit_type() -> Result<(), ParseError> {
        let i = "()".char_indices().collect::<Vec<(usize, char)>>();
        let mut input = Input::new(&i);
        let u = parse_type(&mut input)?;

        match u {
            Type::Unit => Ok(()),
            _ => panic!("should be unit type"),
        }
    }

    #[test]
    fn should_parse_tuple_type() -> Result<(), ParseError> {
        let i = "(alpha, beta, gamma)".char_indices().collect::<Vec<(usize, char)>>();
        let mut input = Input::new(&i);
        let u = parse_type(&mut input)?;

        let types = match u {
            Type::Tuple(ts) => ts, 
            _ => panic!("should be tuple type"),
        };

        assert_eq!( types.len(), 3 );

        let t1_name = match &types[0] {
            Type::Simple(PSym { value, .. }) => value,
            _ => panic!("t1 should be simple type"),
        };

        assert_eq!( t1_name, "alpha" );
        Ok(())
    }

    #[test]
    fn should_parse_arrow_type() -> Result<(), ParseError> {
        let i = "fun(alpha) -> beta".char_indices().collect::<Vec<(usize, char)>>();
        let mut input = Input::new(&i);
        let u = parse_type(&mut input)?;

        let (mut input, output) = match u {
            Type::Fun { input, output } => (input, output), 
            _ => panic!("should be arrow type"),
        };

        assert_eq!( input.len(), 1 );

        let i_name = match input.pop().unwrap() {
            Type::Simple(PSym { value, .. }) => value,
            _ => panic!("input type should be simple"),
        };

        assert_eq!( i_name, "alpha" );

        let o_name = match *output {
            Type::Simple(PSym { value, .. }) => value,
            _ => panic!("input type should be simple"),
        };

        assert_eq!( o_name, "beta" );
        Ok(())
    }

    #[test]
    fn should_parse_paren_type() -> Result<(), ParseError> {
        let i = "(((alpha)))".char_indices().collect::<Vec<(usize, char)>>();
        let mut input = Input::new(&i);
        let u = parse_type(&mut input)?;

        let name = match u {
            Type::Simple(PSym { value, .. }) => value, 
            _ => panic!("should be simple type"),
        };

        assert_eq!( name, "alpha" );
        Ok(())
    }

    #[test]
    fn should_parse_arrow_past_arrow_parameter() -> Result<(), ParseError> {
        let i = "fun(a) -> fun(fun(b) -> c) -> d".char_indices().collect::<Vec<(usize, char)>>();
        let mut input = Input::new(&i);
        let u = parse_type(&mut input)?;


        let (mut input_a, output_bc_etc) = match u {
            Type::Fun {input, output} => (input, *output),
            x => panic!("should be arrow type, but found: {:?}", x),
        };

        assert_eq!( input_a.len(), 1 );

        let name = match input_a.pop().unwrap() {
            Type::Simple( PSym { value, .. }) => value,
            x => panic!("first input should be simple type, but found: {:?}", x),
        };

        assert_eq!( name, "a" );

        let (mut input_bc, output_d) = match output_bc_etc {
            Type::Fun {input, output} => (input, *output),
            x => panic!("first output should be arrow type, but found: {:?}", x),
        };

        assert_eq!( input_bc.len(), 1 );

        let (mut input_b, output_c) = match input_bc.pop().unwrap() {
            Type::Fun {input, output} => (input, *output),
            x => panic!("second input should be arrow type, but found {:?}", x),
        };

        assert_eq!( input_b.len(), 1 );

        let name = match input_b.pop().unwrap() {
            Type::Simple(PSym { value, .. }) => value,
            x => panic!("second input input should be simple type, but found {:?}", x),
        };

        assert_eq!( name, "b" );

        let name = match output_c {
            Type::Simple(PSym { value, .. }) => value,
            x => panic!("second input output should be simple type, but found {:?}", x),
        };

        assert_eq!( name, "c" );

        let name = match output_d {
            Type::Simple(PSym { value, .. }) => value,
            x => panic!("final output should be simple type, but found {:?}", x),
        };

        assert_eq!( name, "d" );

        Ok(())
    }

    #[test]
    fn should_parse_paren_arrows() -> Result<(), ParseError> {
        let i = "fun(a) -> fun(b) -> fun(fun(c) -> d) -> fun( fun( fun(e) -> f ) -> g ) -> i".char_indices().collect::<Vec<(usize, char)>>();
        let mut input = Input::new(&i);
        let u = parse_type(&mut input)?;

        let (mut input_a, output_b_etc) = match u {
            Type::Fun { input, output } => (input, *output), 
            x => panic!("should be arrow type, but found {:?}", x),
        };

        assert_eq!( input_a.len(), 1 );

        let name = match input_a.pop().unwrap() {
            Type::Simple(PSym { value, .. }) => value,
            x => panic!("input_a should be simple type, but found: {:?}", x),
        };

        assert_eq!(name, "a");

        let (mut input_b, output_cd_etc) = match output_b_etc {
            Type::Fun { input, output } => (input, *output),
            x => panic!("input_b_etc should be arrow type, but found: {:?}", x),
        };

        assert_eq!( input_b.len(), 1 );

        let name = match input_b.pop().unwrap() {
            Type::Simple(PSym { value, .. }) => value,
            x => panic!("input_b should be simple type, but found: {:?}", x),
        };
        
        assert_eq!(name, "b");

        let (mut input_cd, output_efg_etc) = match output_cd_etc {
            Type::Fun { input, output } => (input, *output),
            x => panic!("output_cd_etc should be arrow type, but found: {:?}", x),
        };

        assert_eq!( input_cd.len(), 1 );

        let (mut input_c, output_d) = match input_cd.pop().unwrap() {
            Type::Fun { input, output } => (input, *output),
            x => panic!("input_cd should be arrow type, but found: {:?}", x),
        };

        assert_eq!( input_c.len(), 1 );

        let name = match input_c.pop().unwrap() {
            Type::Simple(PSym { value, .. }) => value,
            x => panic!("input_c should be simple type, but found {:?}", x),
        };

        assert_eq!( name, "c" );

        let name = match output_d {
            Type::Simple(PSym { value, .. }) => value,
            x => panic!("output_d should be simple type, but found {:?}", x),
        };

        assert_eq!( name, "d" );
        
        let (mut input_efg, output_i) = match output_efg_etc {
            Type::Fun {input, output} => (input, *output),
            x => panic!("input_efg_etc should be arrow type, but found {:?}", x),
        };

        assert_eq!( input_efg.len(), 1 );

        let (mut input_ef, output_g) = match input_efg.pop().unwrap() {
            Type::Fun {input, output} => (input, *output),
            x => panic!("input_efg should be arrow type, but found {:?}", x),
        };

        assert_eq!( input_ef.len(), 1 );

        let (mut input_e, output_f) = match input_ef.pop().unwrap() {
            Type::Fun {input, output} => (input, *output),
            x => panic!("input_ef should be arrow type, but found {:?}", x),
        };

        assert_eq!( input_e.len(), 1 );

        let name = match input_e.pop().unwrap() {
            Type::Simple(PSym { value, .. }) => value,
            x => panic!("input_e should be simple type, but found {:?}", x),
        };

        assert_eq!( name, "e" );

        let name = match output_f {
            Type::Simple(PSym { value, .. }) => value,
            x => panic!("output_f should be simple type, but found {:?}", x),
        };

        assert_eq!( name, "f" );

        let name = match output_g {
            Type::Simple(PSym { value, .. }) => value,
            x => panic!("output_g should be simple type, but found {:?}", x),
        };

        assert_eq!( name, "g" );

        let name = match output_i {
            Type::Simple(PSym { value, .. }) => value,
            x => panic!("output_i should be simple type, but found {:?}", x),
        };

        assert_eq!( name, "i" );

        Ok(())
    }

    #[test]
    fn should_parse_complex_tuple() -> Result<(), ParseError> {
        let i = "(fun(a) -> b, c::d::e, (), i<j,k,l>, (m, n))".char_indices().collect::<Vec<(usize, char)>>();
        let mut input = Input::new(&i);
        let u = parse_type(&mut input)?;

        let mut types = match u {
            Type::Tuple(types) => types, 
            _ => panic!("should be tuple type"),
        };

        assert_eq!( types.len(), 5 );

        let one = types.remove(0);

        let (mut one_input, one_output) = match one {
            Type::Fun {input, output} => (input, *output),   
            x => panic!("one should be arrow type, but found {:?}", x),
        };

        assert_eq!( one_input.len(), 1 );

        let name = match one_input.pop().unwrap() {
            Type::Simple(PSym { value, .. }) => value,
            x => panic!("one_input should be simple type, but found {:?}", x),
        };

        assert_eq!( name, "a" );

        let name = match one_output {
            Type::Simple(PSym { value, .. }) => value,
            x => panic!("one_output should be simple type, but found {:?}", x),
        };

        assert_eq!( name, "b" );

        let two = types.remove(0);
        
        let (names, t) = match two {
            Type::Namespace(ns, t) => (ns, *t),
            x => panic!("two should be namespace type, but found {:?}", x),
        };
        
        assert_eq!( names.len(), 2 );
        assert_eq!( sym_proj(&names[0]), "c" );
        assert_eq!( sym_proj(&names[1]), "d" );

        let name = match t {
            Type::Simple(PSym { value, .. }) => value,
            x => panic!("t should be simple type, but found {:?}", x),
        };

        assert_eq!( name, "e" );

        let three = types.remove(0);

        assert_eq!( matches!( three, Type::Unit ), true );

        let four = types.remove(0);

        let (name, mut ts) = match four {
            Type::Index(PSym { value, .. }, ts) => (value, ts),
            x => panic!("four should be indexed type, but found {:?}", x),
        };

        assert_eq!( name, "i" );

        assert_eq!( ts.len(), 3 );

        let index_one = ts.remove(0);

        let name = match index_one {
            Type::Simple(PSym { value, .. }) => value,
            x => panic!( "index_one should be simple type, but found {:?}", x),
        };

        assert_eq!( name, "j" );

        let index_two = ts.remove(0);

        let name = match index_two {
            Type::Simple(PSym { value, .. }) => value,
            x => panic!( "index_two should be simple type, but found {:?}", x),
        };

        assert_eq!( name, "k" );

        let index_three = ts.remove(0);

        let name = match index_three {
            Type::Simple(PSym { value, .. }) => value,
            x => panic!( "index_three should be simple type, but found {:?}", x),
        };

        assert_eq!( name, "l" );

        let five = types.remove(0);

        let mut ts = match five {
            Type::Tuple(ts) => ts,
            x => panic!( "five should be tuple type, but found {:?}", x),
        };

        assert_eq!( ts.len(), 2 );
        
        let tuple_one = ts.remove(0);

        let name = match tuple_one {
            Type::Simple(PSym { value, .. }) => value,
            x => panic!( "tuple_one should be simple type but found {:?}", x),
        };

        assert_eq!( name, "m" );

        let tuple_two = ts.remove(0);

        let name = match tuple_two {
            Type::Simple(PSym { value, .. }) => value,
            x => panic!( "tuple_two should be simple type but found {:?}", x),
        };

        assert_eq!( name, "n" );
        Ok(())
    }

    #[test]
    fn should_parse_index_namespace() -> Result<(), ParseError> {
        let i = "a::e<f>".char_indices().collect::<Vec<(usize, char)>>();
        let mut input = Input::new(&i);
        let u = parse_type(&mut input)?;
        
        let (names, t) = match u {
            Type::Namespace(names, t) => (names, *t),
            x => panic!("should be namespace type, but found {:?}", x),
        };

        assert_eq!( names.len(), 1 );
        assert_eq!( sym_proj(&names[0]), "a" );

        let (name, mut ts) = match t {
            Type::Index(PSym { value, .. }, ts) => (value, ts),
            x => panic!("t should be indexed type, but found {:?}", x),
        };

        assert_eq!( name, "e" );

        assert_eq!( ts.len(), 1 );
        
        let index_one = ts.remove(0);

        let name = match index_one {
            Type::Simple(PSym { value, .. }) => value,
            x => panic!("index_one should be simple type, but found {:?}", x),
        };

        assert_eq!( name, "f" );

        Ok(())
    }

    #[test]
    fn should_parse_indexed_arrow_param() -> Result<(), ParseError> {
        let i = "fun(a<b>) -> c<d>".char_indices().collect::<Vec<(usize, char)>>();
        let mut input = Input::new(&i);
        let u = parse_type(&mut input)?;

        let (mut input, output) = match u {
            Type::Fun {input, output} => (input, *output),
            x => panic!("should be arrow type, but found {:?}", x),
        };

        assert_eq!( input.len(), 1 );

        let (name, mut ts) = match input.pop().unwrap() {
            Type::Index(PSym { value, .. }, ts) => (value, ts),
            x => panic!("input should be index type, but found {:?}", x),
        };

        assert_eq!( name, "a" );

        let index_one = ts.remove(0);

        let name = match index_one {
            Type::Simple(PSym { value, .. }) => value,
            x => panic!("index_one should be index type, but found {:?}", x),
        };

        assert_eq!( name, "b" );

        let (name, mut ts) = match output {
            Type::Index(PSym { value, .. }, ts) => (value, ts),
            x => panic!("output should be index type, but found {:?}", x),
        };

        assert_eq!( name, "c" );

        let index_one = ts.remove(0);

        let name = match index_one {
            Type::Simple(PSym { value, .. }) => value,
            x => panic!("index_one should be index type, but found {:?}", x),
        };

        assert_eq!( name, "d" );

        Ok(())
    }

    #[test]
    fn should_parse_namespace_arrow_param() -> Result<(), ParseError> {
        let i = "fun(a::b) -> c::d".char_indices().collect::<Vec<(usize, char)>>();
        let mut input = Input::new(&i);
        let u = parse_type(&mut input)?;

        let (mut input_ab, output_cd) = match u {
            Type::Fun {input, output} => (input, *output),
            x => panic!("should be arrow type, but found {:?}", x),
        };

        assert_eq!( input_ab.len(), 1 );

        let (names, t) = match input_ab.pop().unwrap() {
            Type::Namespace(ns, t) => (ns, *t),
            x => panic!("input_ab should be indexed type, but found {:?}", x),
        };

        assert_eq!( names.len(), 1 );
        assert_eq!( sym_proj(&names[0]), "a" );

        let name = match t {
            Type::Simple(PSym { value, .. }) => value,
            x => panic!("t should be simple type, but found {:?}", x),
        };

        assert_eq!( name, "b" );

        let (names, t) = match output_cd {
            Type::Namespace(ns, t) => (ns, *t),
            x => panic!("output_cd should be indexed type, but found {:?}", x),
        };

        assert_eq!( names.len(), 1 );
        assert_eq!( sym_proj(&names[0]), "c" );

        let name = match t {
            Type::Simple(PSym { value, .. }) => value,
            x => panic!("t should be simple type, but found {:?}", x),
        };

        assert_eq!( name, "d" );

        Ok(())
    }

    #[test]
    fn should_parse_fun_with_multiple_inputs() -> Result<(), ParseError> {
        let i = "fun(a::b, a<b>, a, fun(a) -> a) -> d".char_indices().collect::<Vec<(usize, char)>>();
        let mut input = Input::new(&i);
        let u = parse_type(&mut input)?;

        let (input, output) = match u {
            Type::Fun {input, output} => (input, *output),
            x => panic!("should be arrow type, but found {:?}", x),
        };

        assert_eq!( input.len(), 4 );

        match output {
            Type::Simple(_) => (),
            x => panic!("should be simple type, but found {:?}", x),
        }

        Ok(())
    }
}

