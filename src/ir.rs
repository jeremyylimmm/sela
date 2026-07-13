use std::collections::HashSet;
use std::fmt::Write;

pub enum Op {
    Leaf(usize),
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
}

#[derive(Clone, PartialEq)]
pub struct Shape(Vec<usize>);

struct Use {
    node: Node,
    index: usize,
}

pub struct NodeData {
    node: Node,
    op: Op,
    inputs: Vec<Node>,
    uses: Vec<Use>,
    pub shape: Shape,
    pub strides: Vec<usize>,
}

pub struct IR {
    data: Vec<Option<NodeData>>,
    free_list: Vec<Node>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Node {
    index: usize,
    generation: i64,
}

impl IR {
    pub fn new() -> Self {
        Self {
            data: vec![],
            free_list: vec![],
        }
    }

    pub fn create(&mut self, op: Op, inputs: Vec<Node>, shape: Shape, strides: Vec<usize>) -> Node {
        let node = self.free_list.pop().unwrap_or_else(|| {
            let index = self.data.len();
            self.data.push(None);

            Node {
                index,
                generation: 1,
            }
        });

        for (input_index, &input) in inputs.iter().enumerate() {
            self[input].uses.push(Use {
                node,
                index: input_index,
            });
        }

        self.data[node.index] = Some(NodeData {
            node: node,
            op,
            inputs,
            uses: vec![],
            shape,
            strides,
        });

        node
    }

    fn delete(&mut self, mut node: Node) {
        let data = self.data[node.index].take().expect("dead handle");
        assert!(data.node == node, "dead handle");
        node.generation += 1;
        self.free_list.push(node);
    }

    fn roots(&self) -> Vec<Node> {
        self.data
            .iter()
            .flatten()
            .filter_map(|data| {
                if data.uses.is_empty() {
                    Some(data.node)
                } else {
                    None
                }
            })
            .collect()
    }

    fn gv(&self, visited: &mut HashSet<Node>, out: &mut String, node: Node) {
        if visited.contains(&node) {
            return;
        }

        visited.insert(node);

        for &input in &self[node].inputs {
            self.gv(visited, out, input);
            write!(out, "  n{} -> n{};\n", node.index, input.index).unwrap();
        }

        let label = match self[node].op {
            Op::Leaf(len) => format!("leaf<{}>", len),
            Op::Add => format!("add"),
            Op::Sub => format!("sub"),
            Op::Mul => format!("mul"),
            Op::Div => format!("div"),
            Op::MatMul => format!("matmul"),
        };

        write!(
            out,
            "  n{} [shape=\"box\" label=\"{}\"];\n",
            node.index, label
        )
        .unwrap();
    }

    pub fn graphviz(&self) -> String {
        let mut out = String::new();

        write!(out, "digraph G {{\n").unwrap();

        write!(out, "  rankdir=BT;\n").unwrap();

        let mut visited = HashSet::new();

        for root in self.roots() {
            self.gv(&mut visited, &mut out, root);
        }

        write!(out, "}}").unwrap();

        out
    }
}

impl std::ops::Index<Node> for IR {
    type Output = NodeData;

    fn index(&self, index: Node) -> &Self::Output {
        let data = self.data[index.index].as_ref().expect("dead handle");
        assert!(data.node == index, "dead handle");
        data
    }
}

impl std::ops::IndexMut<Node> for IR {
    fn index_mut(&mut self, index: Node) -> &mut Self::Output {
        let data = self.data[index.index].as_mut().expect("dead handle");
        assert!(data.node == index, "dead handle");
        data
    }
}

impl Shape {
    pub fn from_vec(v: Vec<usize>) -> Self {
        Self(v)
    }

    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }

    pub fn dims(&self) -> usize {
        self.0.len()
    }

    pub fn dense_strides(&self) -> Vec<usize> {
        let mut strides = vec![1; self.dims()];

        for i in (0..strides.len() - 1).rev() {
            strides[i] = strides[i + 1] * self[i + 1];
        }

        strides
    }

    pub fn to_vec(self) -> Vec<usize> {
        self.0
    }

    pub fn pointwise_ok(&self, other: &Shape) -> bool {
        self.0 == other.0
    }

    pub fn matmul_ok(&self, other: &Shape) -> Option<Shape> {
        if self.dims() != other.dims() {
            return None;
        }

        let dims = self.dims();

        if dims < 2
            || self[dims - 1] != other[dims - 2]
            || self.0[0..dims - 2] != other.0[0..dims - 2]
        {
            return None;
        }

        let shape = Shape::from_vec(
            self.0[0..dims - 2]
                .iter()
                .chain(&[self[dims - 2], other[dims - 1]])
                .copied()
                .collect(),
        );

        Some(shape)
    }
}

impl std::ops::Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
