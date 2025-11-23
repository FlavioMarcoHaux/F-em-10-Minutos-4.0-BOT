
export async function getOPFSRoot() {
  return await navigator.storage.getDirectory();
}

export async function createOPFSFile(filename: string): Promise<FileSystemFileHandle> {
  const root = await getOPFSRoot();
  return await root.getFileHandle(filename, { create: true });
}

export async function getOPFSFileAsBlob(filename: string): Promise<Blob> {
  const root = await getOPFSRoot();
  try {
    const fileHandle = await root.getFileHandle(filename);
    const file = await fileHandle.getFile();
    return file;
  } catch (e) {
    throw new Error(`File ${filename} not found in OPFS.`);
  }
}

export async function deleteOPFSFile(filename: string): Promise<void> {
  const root = await getOPFSRoot();
  try {
    await root.removeEntry(filename);
  } catch (e) {
    // Ignore if file doesn't exist
  }
}

export async function writeChunkToStream(writable: FileSystemWritableFileStream, data: Uint8Array) {
  await writable.write(data);
}
