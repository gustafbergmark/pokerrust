use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use std::iter::Sum;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

#[repr(align(256))]
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Vector {
    #[serde(with = "BigArray")]
    pub values: [Float; 1326],
}

pub(crate) type Float = f32;

impl Vector {
    pub fn from(v: &[Float]) -> Self {
        Vector {
            values: v.try_into().unwrap(),
        }
    }
    pub fn ones() -> Self {
        Vector {
            values: [1.0; 1326],
        }
    }

    pub fn sum(&self) -> Float {
        self.values.iter().sum::<Float>()
    }
}

impl Default for Vector {
    fn default() -> Self {
        Vector {
            values: [0.0; 1326],
        }
    }
}

impl Index<usize> for Vector {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

impl Mul for Vector {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut values = [0.0; 1326];
        for i in 0..1326 {
            values[i] = self.values[i] * rhs.values[i];
        }
        Vector { values }
    }
}

impl Mul<Float> for Vector {
    type Output = Self;

    fn mul(self, rhs: Float) -> Self::Output {
        let mut values = [0.0; 1326];
        for i in 0..1326 {
            values[i] = self.values[i] * rhs;
        }
        Vector { values }
    }
}

impl MulAssign for Vector {
    fn mul_assign(&mut self, rhs: Self) {
        for i in 0..1326 {
            self.values[i] *= rhs.values[i];
        }
    }
}

impl MulAssign<Float> for Vector {
    fn mul_assign(&mut self, rhs: Float) {
        for i in 0..1326 {
            self.values[i] *= rhs;
        }
    }
}

impl Add for Vector {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut values = [0.0; 1326];
        for i in 0..1326 {
            values[i] = self.values[i] + rhs.values[i];
        }
        Vector { values }
    }
}

impl AddAssign for Vector {
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..1326 {
            self.values[i] += rhs.values[i];
        }
    }
}

impl Sub for Vector {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut values = [0.0; 1326];
        for i in 0..1326 {
            values[i] = self.values[i] - rhs.values[i];
        }
        Vector { values }
    }
}

impl SubAssign for Vector {
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..1326 {
            self.values[i] -= rhs.values[i];
        }
    }
}

impl Div for Vector {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let mut values = [0.0; 1326];
        for i in 0..1326 {
            values[i] = self.values[i] / rhs.values[i];
        }
        Vector { values }
    }
}

impl DivAssign for Vector {
    fn div_assign(&mut self, rhs: Self) {
        for i in 0..1326 {
            self.values[i] /= rhs.values[i];
        }
    }
}
impl Div<Float> for Vector {
    type Output = Self;

    fn div(self, rhs: Float) -> Self::Output {
        let mut values = [0.0; 1326];
        for i in 0..1326 {
            values[i] = self.values[i] / rhs;
        }
        Vector { values }
    }
}

impl Sum for Vector {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Vector::default(), |a, b| a + b)
    }
}
