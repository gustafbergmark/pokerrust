use serde::de::{Error, SeqAccess, Visitor};
use serde::ser::SerializeTuple;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

#[repr(align(256))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vector<const M: usize> {
    pub values: [f32; M],
}

impl<const M: usize> Serialize for Vector<M> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_tuple(M)?;
        for elem in &self.values[..] {
            seq.serialize_element(elem)?;
        }
        seq.end()
    }
}

impl<'de, const M: usize> Deserialize<'de> for Vector<M> {
    fn deserialize<D>(deserializer: D) -> Result<Vector<M>, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ArrayVisitor<T, const M: usize> {
            element: PhantomData<T>,
        }

        impl<'de, T, const M: usize> Visitor<'de> for ArrayVisitor<T, M>
        where
            T: Default + Copy + Deserialize<'de>,
        {
            type Value = [T; M];

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("Todo GBERGMARK")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<[T; M], A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut arr = [T::default(); M];
                for i in 0..M {
                    arr[i] = seq
                        .next_element()?
                        .ok_or_else(|| Error::invalid_length(i, &self))?;
                }
                Ok(arr)
            }
        }

        let visitor = ArrayVisitor {
            element: PhantomData,
        };
        deserializer
            .deserialize_tuple(M, visitor)
            .map(|arr| Vector { values: arr })
    }
}

impl<const M: usize> Vector<M> {
    pub fn from(v: &[f32]) -> Self {
        Vector {
            values: v.try_into().unwrap(),
        }
    }
    pub fn ones() -> Self {
        Vector { values: [1.0; M] }
    }

    pub fn sum(&self) -> f32 {
        self.values.iter().sum::<f32>()
    }
}

impl<const M: usize> Default for Vector<M> {
    fn default() -> Self {
        Vector { values: [0.0; M] }
    }
}

impl<const M: usize> Index<usize> for Vector<M> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl<const M: usize> IndexMut<usize> for Vector<M> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

impl<const M: usize> Mul for Vector<M> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut values = [0.0; M];
        for i in 0..M {
            values[i] = self.values[i] * rhs.values[i];
        }
        Vector { values }
    }
}

impl<const M: usize> Mul<f32> for Vector<M> {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut values = [0.0; M];
        for i in 0..M {
            values[i] = self.values[i] * rhs;
        }
        Vector { values }
    }
}

impl<const M: usize> MulAssign for Vector<M> {
    fn mul_assign(&mut self, rhs: Self) {
        for i in 0..M {
            self.values[i] *= rhs.values[i];
        }
    }
}

impl<const M: usize> MulAssign<f32> for Vector<M> {
    fn mul_assign(&mut self, rhs: f32) {
        for i in 0..M {
            self.values[i] *= rhs;
        }
    }
}

impl<const M: usize> Add for Vector<M> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut values = [0.0; M];
        for i in 0..M {
            values[i] = self.values[i] + rhs.values[i];
        }
        Vector { values }
    }
}

impl<const M: usize> AddAssign for Vector<M> {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..M {
            self.values[i] += rhs.values[i];
        }
    }
}

impl<const M: usize> Sub for Vector<M> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut values = [0.0; M];
        for i in 0..M {
            values[i] = self.values[i] - rhs.values[i];
        }
        Vector { values }
    }
}

impl<const M: usize> SubAssign for Vector<M> {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..M {
            self.values[i] -= rhs.values[i];
        }
    }
}

impl<const M: usize> Div for Vector<M> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut values = [0.0; M];
        for i in 0..M {
            values[i] = self.values[i] / rhs.values[i];
        }
        Vector { values }
    }
}

impl<const M: usize> DivAssign for Vector<M> {
    fn div_assign(&mut self, rhs: Self) {
        for i in 0..M {
            self.values[i] /= rhs.values[i];
        }
        for i in 0..M {
            if self.values[i].is_nan() {
                self.values[i] = 0.5;
            }
        }
    }
}
impl<const M: usize> Div<f32> for Vector<M> {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        let mut values = [0.0; M];
        for i in 0..M {
            values[i] = self.values[i] / rhs;
        }
        Vector { values }
    }
}
