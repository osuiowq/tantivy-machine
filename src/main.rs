// # Iterating docs and positioms.
//
// At its core of tantivy, relies on a data structure
// called an inverted index.
//
// This example shows how to manually iterate through
// the list of documents containing a term, getting
// its term frequency, and accessing its positions.

// ---
// Importing tantivy...
#[macro_use]
extern crate tantivy;
extern crate rulinalg;
use tantivy::schema::*;
use tantivy::Index;
use tantivy::{DocId, DocSet, Postings};
use tantivy::SegmentWriter;
use tantivy::postings::serializer::*;
use tantivy::postings::MultiFieldPostingsWriter;
use tantivy::SegmentMeta;




use std::collections::HashMap;

use rulinalg::matrix::{BaseMatrixMut, BaseMatrix, Matrix};
use rulinalg::matrix::decomposition::FullPivLu;
use std::str;

extern crate prettytable;
use prettytable::{Table, Row, Cell};

fn main() {
    // Get result rows from tantivy.
    let (mut index, field) = create_test_index().unwrap();
    let tantivy_result = get_tantivy_matrix(index.clone(), field).unwrap();
    
    // get max docid
    let max_docid = tantivy_result.iter().max().unwrap().doc_id.clone();

    // Map terms (strings) to ids (ints).
    let mut terms_map = HashMap::new();
    let mut num_terms = 0;
    for record in &tantivy_result {
        if let Some(_) = terms_map.get(&record.term) {
        } else {
            terms_map.insert(record.term.clone(), num_terms);
            num_terms = num_terms + 1;
        }
    }
    println!("RESULT: {:?}", &tantivy_result);
    let mut ls_matrix = calculate_lsa(tantivy_result, &terms_map, num_terms, max_docid.clone()).unwrap();
    index.load_searchers();
    let searcher = index.searcher();
    let mut total_num_tokens : u64 = 0;
    for segment_reader in searcher.segment_readers() {
        let num_tokens = segment_reader.inverted_index(field).total_num_tokens();
        total_num_tokens = total_num_tokens.clone() + num_tokens;
    }

    let mut new_segment = index.new_segment();
    println!("NEW SEGMENT: {:?}", new_segment.id());
    let mut serializer = InvertedIndexSerializer::open(&mut new_segment).expect("Error: create serializer");;
    let mut field_serializer = serializer.new_field(field.clone(), total_num_tokens).expect("Error: create field_serializer");
    let rows = ls_matrix.col_iter();
    //for mut i  in 0..ls_matrix.cols(){
        //let mut row = rows.next().unwrap();
        field_serializer.new_term(Term::from_field_text(field, &"hallohuhuwehe").as_slice()).expect("Error: new_term");;
        field_serializer.write_doc(DocId::from(0 as u32), 10 , &[]).expect("Error: write_doc");;
        field_serializer.close_term().expect("Error: close term");;
        field_serializer.close().expect("Error: close field_serializer");       
    //}
serializer.close().expect("Error: close serializer");;
index.load_searchers();
new_segment.index();


println!("{:?}",get_segment_ids(index.clone(), field));    
println!("{:?}",get_tantivy_matrix(index.clone(), field).unwrap());    

}

fn calculate_lsa(
    tantivy_result: Vec<TantivyDocTermFreq>, 
    terms_map: &HashMap<String, usize>, 
    num_terms: usize,
    max_docid: u32) -> tantivy::Result<Matrix<f32>> {
    


   

    //println!("terms map {:?}", terms_map);
    //println!("max_docid {:?}", max_docid);
    //println!("num_terms {:?}", num_terms);

    // Create zeroed matrix with docs as columns and terms as rows.
    let mut tf_matrix = Matrix::<f32>::zeros(num_terms, max_docid as usize + 1);

    // Iterate over result rows.
    for record in &tantivy_result {
        let term_index = terms_map.get(&record.term).unwrap();
        println!("record {:?} {:?} {:?}", record.doc_id, *term_index, record.term_freq);

        // Set value in matrix.
        let mut slice = tf_matrix.sub_slice_mut([*term_index, record.doc_id as usize], 1, 1);
        let value = slice.iter_mut().next().unwrap();
        *value = record.term_freq as f32;
    }

        print_term_table(&terms_map, &tf_matrix, "A");
    let lsa_matrix_1 = &tf_matrix * &tf_matrix.transpose();
    let lsa_matrix_2 = &tf_matrix.transpose() * &tf_matrix;
    let (_s_square_1, mut u,u_t) = lsa_matrix_1.clone().svd().unwrap();
    let (_s_square_2, _v, _v_t) = lsa_matrix_2.clone().svd().unwrap();;
        print_matrix_table(_s_square_1, "s");
        print_matrix_table(_v.clone(), "v");
        print_matrix_table(u.clone(), "u");
        print_matrix_table(u_t.clone(), "u_t");

     
    let lu = FullPivLu::decompose(u.clone()).unwrap();
    let rank = lu.rank();
        println!("rank {:#?}", rank);

    // let k = rank - 1;
    let k = 2;
    let t_k = u.sub_slice_mut([0, 0], u.rows(), k) * u.sub_slice_mut([0, 0], u.rows(), k).transpose();
    let tf_matrix_k = &t_k * tf_matrix.clone(); 
    let meow = &tf_matrix_k - &tf_matrix;
    print_term_table(&terms_map, &meow, "A-A_k");
    print_term_table(&terms_map, &tf_matrix, "A");
    print_term_table(&terms_map, &tf_matrix_k, "A_k");
    print_term_table(&terms_map, &meow, "T_k");
    
    Ok(tf_matrix_k)

}
fn print_matrix_table (matrix: Matrix<f32>, message: &str) {
    let mut table = Table::new();
    for row in matrix.row_iter() {
        let mut fields = vec!();
        for val in row.iter() {
            let formatted_val = val;//(val * 100.0).round() / 100.0;
            fields.push(Cell::new(&formatted_val.to_string()));
        }
        table.add_row(Row::new(fields));
    }
    println!("\n{}\n", message);
    table.printstd();
}

fn print_term_table (terms_map: &HashMap<String, usize>, matrix: &Matrix<f32>, message: &str) {
    let mut table = Table::new();
    // let mut header = vec!(Cell::new("id"));
    let mut terms = vec!();
    for (key, _id) in terms_map {
        terms.push(key);
    }
    terms.sort();
    // table.add_row(Row::new(header));
    let mut i = 0;

    for row in matrix.row_iter() {
        let mut fields = vec!();
        // fields.push(Cell::new(&i.to_string()));
        // let term = terms_map[i];
        if i < terms.len() {
            fields.push(Cell::new(terms[i]));
        } else {
            fields.push(Cell::new(""));
        }
        for val in row.iter() {
            let formatted_val = val.round();//(val * 10.0).round();
            fields.push(Cell::new(&formatted_val.to_string()));
        }
        table.add_row(Row::new(fields));
        i = i + 1;
    }
    // Add a row per time
    println!("\n{}\n", message);
    table.printstd();
}

#[derive(Debug, Eq, Ord, PartialEq, PartialOrd)]
struct TantivyDocTermFreq {
    doc_id: DocId,
    term_freq: u32,
    // text: &'a [u8]
    term: String
    // term: Term
}
/* #[derive(Debug, Eq, Ord, PartialEq, PartialOrd)]
struct TantivyTermMap {
    term_map: HashMap<String, usize>,
    term_freq: u32,
    // text: &'a [u8]
    term: String
    // term: Term
}
 */

fn create_test_index<'a>() ->  tantivy::Result<(Index, Field)> {
    // We first create a schema for the sake of the
    // example. Check the `basic_search` example for more information.
    let mut schema_builder = Schema::builder();

    // For this example, we need to make sure to index positions for our title
    // field. `TEXT` precisely does this.
    let title = schema_builder.add_text_field("title", TEXT | STORED);
    let schema = schema_builder.build();
    

    let index = Index::create_in_ram(schema.clone());

    let mut index_writer = index.writer_with_num_threads(1, 50_000_000)?;
    index_writer.add_document(doc!(title => "internet web surfing"));
    index_writer.add_document(doc!(title => "internet surfing"));
    index_writer.add_document(doc!(title => "web surfing"));
    index_writer.add_document(doc!(title => "internet web surfing surfing beach"));
    index_writer.add_document(doc!(title => "surfing beach"));
    index_writer.add_document(doc!(title => "surfing beach"));


    index_writer.commit()?;
    Ok((index, title))
}
fn get_segment_ids(index : Index, field: Field) -> Vec<tantivy::SegmentId> {
    let mut result = Vec::new();
    index.load_searchers();
    let searcher = index.searcher();
    for segment_reader in searcher.segment_readers() {
        result.push(segment_reader.segment_id());
        }

result
}

fn get_tantivy_matrix<'a>(mut index: Index, field: Field) ->  tantivy::Result<Vec<TantivyDocTermFreq>> {
    
    let mut records = vec!();
    index.load_searchers();
    let searcher = index.searcher();

    // A tantivy index is actually a collection of segments.
    // Similarly, a searcher just wraps a list `segment_reader`.
    //
    // (Because we indexed a very small number of documents over one thread
    // there is actually only one segment here, but let's iterate through the list
    // anyway)
    for segment_reader in searcher.segment_readers() {
        // A segment contains different data structure.
        // Inverted index stands for the combination of
        // - the term dictionary
        // - the inverted lists associated to each terms and their positions
        let inverted_index = segment_reader.inverted_index(field);

        // let terms = inverted_index.termdict.
        // println!("termdict {:?}", inverted_index.terms());
        let mut terms = inverted_index.terms().stream();
        // Iterate over the list of terms.
        while terms.advance() {
            // Get current term value.
            let current_term = terms.value();
            // Get current term as string.
            let current_text = str::from_utf8(terms.key()).unwrap().clone();

            // This segment posting object is like a cursor over the documents matching the term.
            // The `IndexRecordOption` arguments tells tantivy we will be interested in both term frequencies
            // and positions.
            //
            // If you don't need all this information, you may get better performance by decompressing less
            // information.
            let mut segment_postings =
                inverted_index.read_postings_from_terminfo(&current_term, IndexRecordOption::WithFreqsAndPositions);

                // this buffer will be used to request for positions
                // let mut positions: Vec<u32> = Vec::with_capacity(100);
                while segment_postings.advance() {
                    // the number of time the term appears in the document.
                    let doc_id: DocId = segment_postings.doc(); //< do not try to access this before calling advance once.

                    // This MAY contains deleted documents as well.
                    if segment_reader.is_deleted(doc_id) {
                        continue;
                    }

                    // the number of time the term appears in the document.
                    let term_freq: u32 = segment_postings.term_freq();

                    // accessing positions is slightly expensive and lazy, do not request
                    // for them if you don't need them for some documents.
                    // segment_postings.positions(&mut positions);

                    // By definition we should have `term_freq` positions.
                    // assert_eq!(positions.len(), term_freq as usize);

                    // Add a record with doc id, term, and term freq.
                    let record = TantivyDocTermFreq {
                        doc_id,
                        term_freq,
                        term: String::from(current_text)
                    };
                    records.push(record);
                }
        }
    }

    Ok(records)
}
