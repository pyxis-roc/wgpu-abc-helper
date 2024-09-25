pub use ffi_support::FfiStr;
use lazy_static::lazy_static;
pub use std::ffi::CStr;
use std::sync::RwLock;
use std::{collections::HashMap, marker::PhantomData};
use thiserror::Error;

#[derive(Error, Debug)]
#[cfg_attr(cffi, repr(u8))]
pub enum Err {
    #[error("CapacityError")]
    CapacityError = 1,
}

#[repr(transparent)]
#[derive(Hash)]
struct Handle<T> {
    id: usize,
    _phantom: std::marker::PhantomData<T>,
}

// impl<T> Eq for Handle<T> {}

// impl<T> std::hash::Hash for Handle<T> {
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         self.id.hash(state);
//     }
// }
/// A registry that acts as a container for handles to the variables within.
struct Registry<T> {
    handles: HashMap<usize, std::sync::Arc<T>>,
    counter: usize,
    reuse: Vec<usize>,
}

impl<T> Registry<T> {
    fn create_item<V>(&mut self, item: V) -> Result<Handle<T>, Err>
    where
        V: Into<std::sync::Arc<T>>,
    {
        let id = if let Some(reuse) = self.reuse.pop() {
            reuse
        } else {
            if self.counter == usize::MAX {
                return Err(Err::CapacityError);
            }
            self.counter += 1;
            self.counter
        };

        // insert the item into the registry..
        self.handles.insert(id, item.into());

        Ok(Handle {
            id,
            _phantom: PhantomData,
        })
    }

    /// Remove an item from the registry. Returns true if the item was removed.
    fn delete_item(&mut self, item: Handle<T>) -> bool {
        self.handles.remove(&item.id).is_some()
    }
}
pub struct Context {
    helper: super::ConstraintHelper,
    vars: Registry<super::Var>,
    expressions: Registry<super::Expression>,
}

lazy_static! {
    static ref CONTEXT_REGISTRY: RwLock<Registry<Context>> = RwLock::new(Registry {
        handles: HashMap::new(),
        counter: 0,
        reuse: Vec::new(),
    });
    static ref global_context: RwLock<Context> = RwLock::new(Context {
        helper: super::ConstraintHelper::default(),
        vars: HashMap::new(),
        expressions: HashMap::new(),
    });
}

#[repr(transparent)]
pub struct ContextHandle {
    pub id: u64,
}

pub struct VarHandle {
    pub(crate) id: u64,
}

pub struct ExpressionHandle {
    pub(crate) id: u64,
}

/// Contexts are *not* thread safe.
impl ContextHandle {
    fn get_helper(&self) -> &super::ConstraintHelper {
        let contexts = CONTEXT_REGISTRY.read().unwrap();
        contexts.handles.get(&self.id).unwrap()
    }

    fn get_helper_mut(&self) -> &mut super::ConstraintHelper {
        let registry = CONTEXT_REGISTRY.write().unwrap();
        registry.0.get(&self.id).unwrap()
    }

    pub extern "C" fn mark_type(self, varname: FfiStr, ty: FfiStr) -> VarHandle {
        // make a new type
        self.get_helper_mut()
            .mark_type(varname.as_str(), ty.into_string());
        let mut registry = VAR_REGISTRY.write().unwrap();
        let id = registry.1;

        // Increment ID
        registry.0.insert(
            id,
            std::sync::Arc::new(super::mark_type(varname.as_str(), ty.into_string())),
        );
        registry.1 += 1;
        VarHandle { id }
    }
}
impl Context {
    pub extern "C" fn get_constraint_helper() -> *mut super::ConstraintHelper {
        Box::into_raw(Box::new(super::ConstraintHelper::default()))
    }

    #[no_mangle]
    pub extern "C" fn mark_type(varname: FfiStr, ty: FfiStr) -> VarHandle {
        // make a new type
        let mut registry = VAR_REGISTRY.write().unwrap();
        let id = registry.1;

        // Increment ID
        registry.0.insert(
            id,
            std::sync::Arc::new(super::mark_type(varname.as_str(), ty.into_string())),
        );
        registry.1 += 1;
        VarHandle { id }
    }

    /// Mark the length of a dimension
    #[no_mangle]
    pub extern "C" fn mark_length(var: VarHandle, dim: u8, size: u64) {
        let registry = VAR_REGISTRY.read().unwrap();
        let var = registry.0.get(&var.id).unwrap();
        super::mark_length(&var.name, dim, size);
    }
}
