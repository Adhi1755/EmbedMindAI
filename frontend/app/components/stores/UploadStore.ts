import { create } from 'zustand';

interface UploadState {
  uploadedFile: File | null;
  isProcessing: boolean;
  setUploadedFile: (file: File | null) => void;
  setProcessing: (status: boolean) => void;
}

export const useUploadStore = create<UploadState>((set) => ({
  uploadedFile: null,
  isProcessing: false,
  setUploadedFile: (file) => set({ uploadedFile: file }),
  setProcessing: (status) => set({ isProcessing: status }),
}));