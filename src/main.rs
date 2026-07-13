use sela::Graph;

fn main() {
    let graph = Graph::new();

    let a = graph.tensor(vec![3, 2]);
    let b = graph.tensor(vec![2, 3]);
    let _c = a.matmul(b);

    println!("{}", graph.graphviz_ir());
}
