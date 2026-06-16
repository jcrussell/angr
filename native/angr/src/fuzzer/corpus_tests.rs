use super::*;

#[test]
fn test_inmem_corpus_serialization() {
    let corpus =
        PyInMemoryCorpus::from_list(vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]]).unwrap();

    let serialied_state = corpus.__getstate__().unwrap();
    let mut deserialized_corpus = PyInMemoryCorpus::py_new().unwrap();
    deserialized_corpus.__setstate__(serialied_state).unwrap();

    let bytes_list = deserialized_corpus.to_bytes_list().unwrap();
    assert_eq!(bytes_list.len(), 3);
    assert_eq!(bytes_list[0], vec![1, 2, 3]);
    assert_eq!(bytes_list[1], vec![4, 5, 6]);
    assert_eq!(bytes_list[2], vec![7, 8, 9]);
}

#[test]
fn test_ondisk_corpus_serialization() {
    let tempdir = tempfile::tempdir().unwrap();
    let mut corpus = PyOnDiskCorpus::py_new(tempdir.path().to_string_lossy().into()).unwrap();
    corpus.add(vec![1, 2, 3]).unwrap();
    corpus.add(vec![4, 5, 6]).unwrap();
    corpus.add(vec![7, 8, 9]).unwrap();

    let serialied_state = corpus.__getstate__().unwrap();
    let tempdir2 = tempfile::tempdir().unwrap();
    let mut deserialized_corpus =
        PyOnDiskCorpus::py_new(tempdir2.path().to_string_lossy().into()).unwrap();
    deserialized_corpus.__setstate__(serialied_state).unwrap();

    let bytes_list = deserialized_corpus.to_bytes_list().unwrap();

    assert_eq!(bytes_list.len(), 3);
    assert_eq!(bytes_list[0], vec![1, 2, 3]);
    assert_eq!(bytes_list[1], vec![4, 5, 6]);
    assert_eq!(bytes_list[2], vec![7, 8, 9]);
}
