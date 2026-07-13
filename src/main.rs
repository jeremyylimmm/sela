use sela::Graph;

fn main() {
    let graph = Graph::new();

    let a = graph.tensor(vec![2, 3]).transpose(0, 1);
    let b = graph.tensor(vec![2, 3]);
    let _c = a.matmul(b);

    println!("{}", graph.graphviz_ir());
}
