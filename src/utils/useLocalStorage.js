export default function useLocalStorage(key, initialValue) {
  const storedValue = localStorage.getItem(key);

  // string to object.
  const initial = storedValue ? JSON.parse(storedValue) : initialValue;

  const stored = ref(initial);

  watch(
    stored,
    (newVal) => {
      localStorage.setItem(key, JSON.stringify(newVal));
    },
    { deep: true }
  );

  const storeValue = (value) => {
    stored.value = value;
  };

  const checkValue = () => {
    return stored.value;
  };

  return { stored, storeValue, checkValue };
}
