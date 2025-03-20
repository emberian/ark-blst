use crate::fp2::Fp2;
use crate::memory;
use core::hash::Hash;
use core::{
    fmt,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};
use std::cmp::Ordering;

use ark_ff::{AdditiveGroup, BigInteger, SqrtPrecomputation};
use std::ops::{Div, DivAssign};

use ark_ff::BigInt;
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, EmptyFlags, Flags, SerializationError,
    Valid, Validate,
};
use ff::Field;
use num_traits::{One, Zero};
use std::{hash::Hasher, ops::Deref};
use zeroize::Zeroize;

use crate::fp::Fp;

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
#[repr(transparent)]
pub struct Fp6(pub(crate) blstrs::Fp6);

impl Ord for Fp6 {
    #[inline(always)]
    fn cmp(&self, other: &Fp6) -> Ordering {
        match self.c2().cmp(&other.c2()) {
            Ordering::Greater => Ordering::Greater,
            Ordering::Less => Ordering::Less,
            Ordering::Equal => match self.c1().cmp(&other.c1()) {
                Ordering::Greater => Ordering::Greater,
                Ordering::Less => Ordering::Less,
                Ordering::Equal => self.c0().cmp(&other.c0()),
            },
        }
    }
}

impl PartialOrd for Fp6 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Fp6) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Deref for Fp6 {
    type Target = blstrs::Fp6;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for Fp6 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl AddAssign<&Fp6> for Fp6 {
    #[inline]
    fn add_assign(&mut self, rhs: &Fp6) {
        self.0.add_assign(rhs.0)
    }
}

impl AddAssign<Fp6> for Fp6 {
    #[inline]
    fn add_assign(&mut self, rhs: Fp6) {
        self.0.add_assign(rhs.0)
    }
}

impl<'a> AddAssign<&'a mut Fp6> for Fp6 {
    fn add_assign(&mut self, rhs: &'a mut Fp6) {
        self.0.add_assign(rhs.0);
    }
}

impl Add<Fp6> for Fp6 {
    type Output = Fp6;

    #[inline]
    fn add(self, rhs: Fp6) -> Fp6 {
        Fp6(self.0 + rhs.0)
    }
}

impl Add<&Fp6> for &Fp6 {
    type Output = Fp6;

    #[inline]
    fn add(self, rhs: &Fp6) -> Fp6 {
        Fp6(self.0 + rhs.0)
    }
}

impl Add<&Fp6> for Fp6 {
    type Output = Fp6;
    #[inline]
    fn add(self, rhs: &Fp6) -> Fp6 {
        Fp6(self.0 + rhs.0)
    }
}
impl Neg for &Fp6 {
    type Output = Fp6;

    #[inline]
    fn neg(self) -> Fp6 {
        Fp6(self.0.neg())
    }
}

impl Neg for Fp6 {
    type Output = Fp6;

    #[inline]
    fn neg(self) -> Fp6 {
        Fp6(self.0.neg())
    }
}

impl SubAssign<&Fp6> for Fp6 {
    #[inline]
    fn sub_assign(&mut self, rhs: &Fp6) {
        self.0.sub_assign(rhs.0);
    }
}

impl<'a> SubAssign<&'a mut Fp6> for Fp6 {
    fn sub_assign(&mut self, rhs: &'a mut Fp6) {
        self.0.sub_assign(rhs.0);
    }
}

impl SubAssign<Fp6> for Fp6 {
    fn sub_assign(&mut self, rhs: Fp6) {
        self.0.sub_assign(rhs.0)
    }
}

impl Sub<&Fp6> for &Fp6 {
    type Output = Fp6;

    #[inline]
    fn sub(self, rhs: &Fp6) -> Fp6 {
        Fp6(self.0 - rhs.0)
    }
}

impl MulAssign<&Fp6> for Fp6 {
    #[inline]
    fn mul_assign(&mut self, rhs: &Fp6) {
        self.0.mul_assign(rhs.0);
    }
}

impl MulAssign<Fp6> for Fp6 {
    fn mul_assign(&mut self, rhs: Self) {
        self.0.mul_assign(rhs.0);
    }
}

impl<'a> MulAssign<&'a mut Fp6> for Fp6 {
    fn mul_assign(&mut self, rhs: &'a mut Fp6) {
        self.0.mul_assign(rhs.0);
    }
}

impl Mul<&Fp6> for Fp6 {
    type Output = Fp6;

    #[inline]
    fn mul(self, rhs: &Fp6) -> Fp6 {
        Fp6(self.0 * rhs.0)
    }
}

impl Mul<Fp6> for Fp6 {
    type Output = Fp6;
    #[inline]
    fn mul(self, rhs: Fp6) -> Fp6 {
        Fp6(self.0 * rhs.0)
    }
}

impl Mul<&Fp6> for &Fp6 {
    type Output = Fp6;

    #[inline]
    fn mul(self, rhs: &Fp6) -> Fp6 {
        Fp6(self.0 * rhs.0)
    }
}

impl<'a> Mul<&'a mut Fp6> for Fp6 {
    type Output = Fp6;

    fn mul(self, rhs: &'a mut Fp6) -> Self::Output {
        Fp6(self.0 * rhs.0)
    }
}

impl One for Fp6 {
    fn one() -> Self {
        Fp6(blstrs::Fp6::ONE)
    }
}

impl Zero for Fp6 {
    fn zero() -> Self {
        Fp6(blstrs::Fp6::ZERO)
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero().into()
    }
}

impl Zeroize for Fp6 {
    fn zeroize(&mut self) {
        self.0 = blstrs::Fp6::from(0);
    }
}

// TODO check invariant
#[allow(unknown_lints, renamed_and_removed_lints)]
#[allow(clippy::derived_hash_with_manual_eq, clippy::derive_hash_xor_eq)]
impl Hash for Fp6 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(&self.0.to_bytes_le()[..]);
    }
}

impl Valid for Fp6 {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

#[allow(clippy::redundant_closure)]
impl ark_serialize::CanonicalDeserialize for Fp6 {
    fn deserialize_with_mode<R: ark_serialize::Read>(
        mut reader: R,
        _compress: Compress,
        _validate: Validate,
    ) -> Result<Self, SerializationError> {
        let mut buff = [0u8; 288];
        reader
            .read(&mut buff[..])
            .map_err(|_| SerializationError::InvalidData)?;
        Option::from(blstrs::Fp6::from_bytes_le(buff).map(|f| Fp6(f)))
            .ok_or(SerializationError::InvalidData)
    }
}

impl ark_serialize::CanonicalDeserializeWithFlags for Fp6 {
    fn deserialize_with_flags<R: ark_serialize::Read, F: Flags>(
        reader: R,
    ) -> Result<(Self, F), SerializationError> {
        Self::deserialize_with_mode(reader, Compress::No, Validate::No)
            .map(|s| (s, F::from_u8(0).unwrap()))
    }
}

impl ark_serialize::CanonicalSerialize for Fp6 {
    fn serialize_with_mode<W: ark_serialize::Write>(
        &self,
        mut writer: W,
        _compress: Compress,
    ) -> Result<(), SerializationError> {
        writer
            .write(&self.0.to_bytes_le()[..])
            .map(|_| ())
            .map_err(|_| SerializationError::InvalidData)
    }

    fn serialized_size(&self, _compress: Compress) -> usize {
        288
    }
}

impl ark_serialize::CanonicalSerializeWithFlags for Fp6 {
    fn serialize_with_flags<W: ark_serialize::Write, F: Flags>(
        &self,
        writer: W,
        _flags: F,
    ) -> Result<(), SerializationError> {
        Self::serialize_with_mode(self, writer, Compress::No)
    }

    fn serialized_size_with_flags<F: Flags>(&self) -> usize {
        Self::serialized_size(self, Compress::No)
    }
}

impl From<bool> for Fp6 {
    fn from(value: bool) -> Self {
        match value {
            true => Fp6::from(1u64),
            false => Fp6::from(0u64),
        }
    }
}

macro_rules! impl_from {
    ($t:ty) => {
        impl From<$t> for Fp6 {
            fn from(value: $t) -> Self {
                Fp6(blstrs::Fp6::from(value as u64))
            }
        }
    };
}

impl_from!(u8);
impl_from!(u16);
impl_from!(u32);
impl_from!(u64);
impl_from!(i8);
impl_from!(i16);
impl_from!(i32);
impl_from!(i64);

impl From<u128> for Fp6 {
    fn from(value: u128) -> Self {
        Fp6::new(Fp2::from(value), Fp2::zero(), Fp2::zero())
    }
}

impl From<i128> for Fp6 {
    fn from(value: i128) -> Self {
        Fp6::new(Fp2::from(value), Fp2::zero(), Fp2::zero())
    }
}



impl<'a> core::iter::Product<&'a Fp6> for Fp6 {
    fn product<I: Iterator<Item = &'a Fp6>>(iter: I) -> Self {
        iter.fold(Fp6::one(), |acc, x| acc * x)
    }
}

impl core::iter::Product<Fp6> for Fp6 {
    fn product<I: Iterator<Item = Fp6>>(iter: I) -> Self {
        iter.fold(Fp6::one(), |acc, x| acc * x)
    }
}

impl<'a> core::iter::Sum<&'a Fp6> for Fp6 {
    fn sum<I: Iterator<Item = &'a Fp6>>(iter: I) -> Self {
        iter.fold(Fp6::zero(), |acc, x| acc + x)
    }
}

impl core::iter::Sum<Fp6> for Fp6 {
    fn sum<I: Iterator<Item = Fp6>>(iter: I) -> Self {
        iter.fold(Fp6::zero(), |acc, x| acc + x)
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl Div<Fp6> for Fp6 {
    type Output = Fp6;

    fn div(self, rhs: Fp6) -> Self::Output {
        self * Fp6(rhs.0.invert().unwrap())
    }
}

impl<'a> Div<&'a mut Fp6> for Fp6 {
    type Output = Fp6;

    fn div(self, rhs: &'a mut Fp6) -> Self::Output {
        let mut c = self;
        c.div_assign(rhs);
        c
    }
}
impl<'a> Div<&'a Fp6> for Fp6 {
    type Output = Fp6;

    fn div(self, rhs: &'a Fp6) -> Self::Output {
        let mut c = self;
        c.div_assign(rhs);
        c
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl DivAssign for Fp6 {
    fn div_assign(&mut self, rhs: Self) {
        *self *= Fp6(rhs.0.invert().unwrap());
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl<'a> DivAssign<&'a Fp6> for Fp6 {
    fn div_assign(&mut self, rhs: &'a Fp6) {
        *self *= Fp6(rhs.0.invert().unwrap());
    }
}

#[allow(clippy::suspicious_op_assign_impl)]
impl<'a> DivAssign<&'a mut Fp6> for Fp6 {
    fn div_assign(&mut self, rhs: &'a mut Fp6) {
        *self *= Fp6(rhs.0.invert().unwrap());
    }
}

impl<'a> Add<&'a mut Fp6> for Fp6 {
    type Output = Fp6;

    fn add(self, rhs: &'a mut Fp6) -> Self::Output {
        Fp6(self.0 + rhs.0)
    }
}

impl Sub<Fp6> for Fp6 {
    type Output = Fp6;

    fn sub(self, rhs: Fp6) -> Self::Output {
        Fp6(self.0 - rhs.0)
    }
}

impl<'a> Sub<&'a Fp6> for Fp6 {
    type Output = Fp6;

    fn sub(self, rhs: &'a Fp6) -> Self::Output {
        Fp6(self.0 - rhs.0)
    }
}

impl<'a> Sub<&'a mut Fp6> for Fp6 {
    type Output = Fp6;

    fn sub(self, rhs: &'a mut Fp6) -> Self::Output {
        Fp6(self.0 - rhs.0)
    }
}

impl ark_ff::UniformRand for Fp6 {
    fn rand<R: ark_std::rand::Rng + ?Sized>(rng: &mut R) -> Self {
        Fp6(blstrs::Fp6::random(rng))
    }
}
impl From<num_bigint::BigUint> for Fp6 {
    fn from(value: num_bigint::BigUint) -> Self {
        Fp6(
            blstrs::Fp6::from_bytes_le(*memory::slice_to_constant_size(&value.to_bytes_le()))
                .unwrap(),
        )
    }
}

impl From<BigInt<12>> for Fp6 {
    fn from(value: BigInt<12>) -> Self {
        Fp6(
            blstrs::Fp6::from_bytes_le(*memory::slice_to_constant_size(&value.to_bytes_le()))
                .unwrap(),
        )
    }
}

impl From<ark_bls12_381::Fq2> for Fp6 {
    fn from(value: ark_bls12_381::Fq2) -> Self {
        let mut buff = Vec::with_capacity(288);
        value.serialize_compressed(&mut buff).unwrap();
        Fp6(blstrs::Fp6::from_bytes_le(*memory::slice_to_constant_size(&buff)).unwrap())
    }
}
impl From<Fp6> for num_bigint::BigUint {
    fn from(value: Fp6) -> Self {
        let slice = value.0.to_bytes_le();
        let s = memory::constant_size_to_slice(&slice);
        num_bigint::BigUint::from_bytes_le(s)
    }
}

impl Fp6 {
    pub const fn new(c0: Fp2, c1: Fp2, c2: Fp2) -> Self {
        Fp6(blstrs::Fp6::new(c0.0, c1.0, c2.0))
    }
}

impl AdditiveGroup for Fp6 {
    type Scalar = Fp6;

    const ZERO: Self = Fp6::new(Fp2::ZERO, Fp2::ZERO, Fp2::ZERO);
}

impl ark_ff::Field for Fp6 {
    type BasePrimeField = crate::fp::Fp;

    const SQRT_PRECOMP: Option<SqrtPrecomputation<Fp6>> = None;

    const ONE: Self = Fp6::new(
        Fp2(blstrs::Fp2::new(blstrs::fp::R, blstrs::Fp::ZERO)),
        Fp2::ZERO,
        Fp2::ZERO,
    );

    fn extension_degree() -> u64 {
        Self::BasePrimeField::extension_degree() * 6
    }

    fn to_base_prime_field_elements(&self) -> impl Iterator<Item = Self::BasePrimeField> {
        Fp2(self.0.c0())
            .to_base_prime_field_elements()
            .chain(Fp2(self.0.c1()).to_base_prime_field_elements())
            .chain(Fp2(self.0.c2()).to_base_prime_field_elements())
            .collect::<Vec<_>>()
            .into_iter()
    }

    fn from_base_prime_field_elems(elems: impl IntoIterator<Item = Self::BasePrimeField>) -> Option<Self> {
        let mut elems = elems.into_iter();
        let elems = elems.by_ref();
        let base_ext_deg = Fp2::extension_degree() as usize;
        let element = Some(Self::new(
            Fp2::from_base_prime_field_elems(elems.take(base_ext_deg))?,
            Fp2::from_base_prime_field_elems(elems.take(base_ext_deg))?,
            Fp2::from_base_prime_field_elems(elems.take(base_ext_deg))?,
        ));
        if elems.next().is_some() {
            None
        } else {
            element
        }
    }

    fn from_base_prime_field(elem: Self::BasePrimeField) -> Self {
        Self::new(Fp2::from_base_prime_field(elem), Fp2::ZERO, Fp2::ZERO)
    }

    fn from_random_bytes_with_flags<F: Flags>(_bytes: &[u8]) -> Option<(Self, F)> {
        unimplemented!()
        //let blst_buffer: &[u8; 288] = memory::slice_to_constant_size(bytes);
        //blstrs::Fp6::from_bytes_le(blst_buffer)
        //    .map(|fp| (Fp6(fp), F::from_u8(0).unwrap()))
        //    .into()
    }

    fn legendre(&self) -> ark_ff::LegendreSymbol {
        todo!()
    }

    fn square(&self) -> Self {
        Fp6(self.0.square())
    }

    fn square_in_place(&mut self) -> &mut Self {
        self.0 = self.0.square();
        self
    }

    #[allow(clippy::redundant_closure)]
    fn inverse(&self) -> Option<Self> {
        self.0.invert().map(|f| Fp6(f)).into()
    }

    fn inverse_in_place(&mut self) -> Option<&mut Self> {
        self.0
            .invert()
            .map(|f| {
                self.0 = f;
                self
            })
            .into()
    }

    // nothing on base prime field
    fn frobenius_map_in_place(&mut self, _power: usize) {}

    fn characteristic() -> &'static [u64] {
        blstrs::fp::MODULUS[..].as_ref()
    }

    fn from_random_bytes(bytes: &[u8]) -> Option<Self> {
        Self::from_random_bytes_with_flags::<EmptyFlags>(bytes).map(|f| f.0)
    }

    #[allow(clippy::redundant_closure)]
    fn sqrt(&self) -> Option<Self> {
        match Self::SQRT_PRECOMP {
            Some(tv) => tv.sqrt(self),
            None => self.0.sqrt().map(|f| Fp6(f)).into(),
        }
    }

    fn sqrt_in_place(&mut self) -> Option<&mut Self> {
        unimplemented!("sqrt_in_place not implemented for Fp6")
    }

    fn sum_of_products<const T: usize>(a: &[Self; T], b: &[Self; T]) -> Self {
        let mut sum = Self::zero();
        for i in 0..a.len() {
            sum += a[i] * b[i];
        }
        sum
    }

    fn frobenius_map(&self, power: usize) -> Self {
        let mut this = *self;
        this.frobenius_map_in_place(power);
        this
    }

    fn pow<S: AsRef<[u64]>>(&self, exp: S) -> Self {
        Fp6(self.0.pow_vartime(exp.as_ref()))
    }

    fn pow_with_table<S: AsRef<[u64]>>(powers_of_2: &[Self], exp: S) -> Option<Self> {
        let mut res = Self::one();
        for (pow, bit) in ark_ff::BitIteratorLE::without_trailing_zeros(exp).enumerate() {
            if bit {
                res *= powers_of_2.get(pow)?;
            }
        }
        Some(res)
    }

    fn mul_by_base_prime_field(&self, elem: &Self::BasePrimeField) -> Self {
        fn conv(f: blstrs::Fp2) -> (blstrs::Fp, blstrs::Fp) {
            (f.c0(), f.c1())
        }

        let mul_elem = |e: blstrs::Fp2| -> Fp2 {
            let (c0, c1) = conv(e);
            Fp2::new(Fp(c0 * elem.0), Fp(c1 * elem.0))
        };
        Self::new(
            mul_elem(self.0.c0()),
            mul_elem(self.0.c1()),
            mul_elem(self.0.c2()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp6_tests() {
        crate::tests::field_test::<Fp6>();
    }
}
