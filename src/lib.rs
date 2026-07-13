pub mod ir;

use ir::{IR, Op};
use std::cell::{RefCell, RefMut};

pub use ir::Shape;

pub struct Graph {
    ir: RefCell<ir::IR>,
}

pub struct Node<'a> {
    inner: ir::Node,
    graph: &'a Graph,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            ir: RefCell::new(IR::new()),
        }
    }

    fn ir(&self) -> RefMut<'_, IR> {
        self.ir.borrow_mut()
    }

    fn create(&self, op: Op, inputs: Vec<ir::Node>, shape: Shape, strides: Vec<usize>) -> Node<'_> {
        let inner = self.ir().create(op, inputs, shape, strides);

        Node { inner, graph: self }
    }

    pub fn tensor(&self, shape: Vec<usize>) -> Node<'_> {
        let shape = Shape::from_vec(shape);
        let strides = shape.dense_strides();
        self.create(Op::Leaf(shape.numel()), vec![], shape, strides)
    }

    pub fn graphviz_ir(&self) -> String {
        self.ir().graphviz()
    }
}

impl<'a> Node<'a> {
    fn shape(&self) -> Shape {
        self.graph.ir()[self.inner].shape.clone()
    }

    pub fn dims(&self) -> usize {
        self.graph.ir()[self.inner].shape.dims()
    }

    fn pointwise(&self, rhs: Node, op: Op) -> Node<'a> {
        let shape = self.shape();
        let strides = shape.dense_strides();
        assert!(shape == rhs.shape());
        self.graph
            .create(op, vec![self.inner, rhs.inner], shape, strides)
    }

    pub fn matmul(&self, rhs: Node) -> Node<'a> {
        let shape = self
            .shape()
            .matmul_ok(&rhs.shape())
            .expect("invalid shapes for matmul");
        let strides = shape.dense_strides();

        self.graph
            .create(Op::MatMul, vec![self.inner, rhs.inner], shape, strides)
    }
}

impl<'a> std::ops::Add<Node<'a>> for Node<'a> {
    type Output = Node<'a>;

    fn add(self, rhs: Node) -> Self::Output {
        self.pointwise(rhs, Op::Add)
    }
}

impl<'a> std::ops::Sub<Node<'a>> for Node<'a> {
    type Output = Node<'a>;

    fn sub(self, rhs: Node) -> Self::Output {
        self.pointwise(rhs, Op::Sub)
    }
}

impl<'a> std::ops::Mul<Node<'a>> for Node<'a> {
    type Output = Node<'a>;

    fn mul(self, rhs: Node) -> Self::Output {
        self.pointwise(rhs, Op::Mul)
    }
}

impl<'a> std::ops::Div<Node<'a>> for Node<'a> {
    type Output = Node<'a>;

    fn div(self, rhs: Node) -> Self::Output {
        self.pointwise(rhs, Op::Div)
    }
}
