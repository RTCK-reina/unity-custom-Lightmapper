using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;
using Random = UnityEngine.Random;

namespace RTCK.NeuraBake.Runtime
{
    public class BakingCore : ILightmapRenderer
    {
        // RenderTexture ���� Texture2D �֕ϊ����郁�\�b�h
        public static Texture2D ConvertRenderTextureToTexture2D(RenderTexture rt)
        {
            if (rt == null)
            {
                Debug.LogError("RenderTexture is null");
                return null;
            }

            var currentRT = RenderTexture.active;
            try
            {
                RenderTexture.active = rt;
                Texture2D tex = new Texture2D(rt.width, rt.height, TextureFormat.RGBA32, false);
                tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
                tex.Apply();
                return tex;
            }
            catch (Exception e)
            {
                Debug.LogError($"Error converting RenderTexture to Texture2D: {e.Message}");
                return null;
            }
            finally
            {
                RenderTexture.active = currentRT;
            }
        }

        // Texture2D �� PNG �t�@�C���Ƃ��ĕۑ����郁�\�b�h
        public static bool SaveTexture2DToPNG(Texture2D tex, string path)
        {
            if (tex == null)
            {
                Debug.LogError("Texture2D is null");
                return false;
            }

            try
            {
                byte[] pngData = tex.EncodeToPNG();
                if (pngData == null)
                {
                    Debug.LogError("Failed to encode texture to PNG");
                    return false;
                }

                File.WriteAllBytes(path, pngData);
                // Debug.Log($"Texture saved to: {path}"); // �E�B���h�E���ŕۑ����b�Z�[�W���o���̂ŏd��������l��
                return true;
            }
            catch (Exception e)
            {
                Debug.LogError($"Error saving PNG file: {e.Message}");
                return false;
            }
        }

        private readonly NeuraBakeSettings settings;
        private readonly Light[] sceneLights;
        private readonly NeuraBakeEmissiveSurface[] emissiveSurfaces;
        private readonly Dictionary<Material, MaterialProperties> materialCache = new Dictionary<Material, MaterialProperties>();
        private readonly Dictionary<Mesh, MeshDataCache> meshDataCache = new Dictionary<Mesh, MeshDataCache>();
        private readonly object cacheLock = new object();

        private class SpatialCache
        {
            private readonly struct RaycastKey : IEquatable<RaycastKey>
            {
                public readonly Vector3 Origin;
                public readonly Vector3 Direction;
                public readonly float MaxDistance;
                private const float OriginPrecisionFactor = 100f;
                private const float DirectionPrecisionFactor = 1000f;
                private const float DistancePrecisionFactor = 100f;

                public RaycastKey(Vector3 origin, Vector3 direction, float maxDistance)
                {
                    Origin = new Vector3(
                        Mathf.Round(origin.x * OriginPrecisionFactor) / OriginPrecisionFactor,
                        Mathf.Round(origin.y * OriginPrecisionFactor) / OriginPrecisionFactor,
                        Mathf.Round(origin.z * OriginPrecisionFactor) / OriginPrecisionFactor
                    );
                    Direction = direction.normalized;
                    MaxDistance = Mathf.Round(maxDistance * DistancePrecisionFactor) / DistancePrecisionFactor;
                }

                public bool Equals(RaycastKey other)
                {
                    return Origin.Equals(other.Origin) &&
                           Direction.Equals(other.Direction) &&
                           MaxDistance == other.MaxDistance;
                }
                public override bool Equals(object obj) => obj is RaycastKey other && Equals(other);
                public override int GetHashCode()
                {
                    unchecked
                    {
                        int hash = 17;
                        hash = hash * 23 + Origin.GetHashCode();
                        hash = hash * 23 + Direction.GetHashCode();
                        hash = hash * 23 + MaxDistance.GetHashCode();
                        return hash;
                    }
                }
            }
            private readonly ConcurrentDictionary<RaycastKey, bool> cache = new ConcurrentDictionary<RaycastKey, bool>();
            public bool Raycast(Ray ray, float maxDistance)
            {
                var key = new RaycastKey(ray.origin, ray.direction, maxDistance);
                return cache.GetOrAdd(key, k => Physics.Raycast(k.Origin, k.Direction, k.MaxDistance));
            }
            public void Clear() { cache.Clear(); }
        }

        private readonly SpatialCache mainRaycastCache = new SpatialCache();

        private class MaterialProperties
        {
            public Material material;
            public Color BaseColor { get; }
            public float Metallic { get; }
            public float Roughness { get; }
            public Texture2D BaseColorMap { get; }
            public Texture2D MetallicGlossMap { get; }
            public Texture2D NormalMap { get; }
            public Vector4 MainTexST { get; }
            public float Smoothness { get; } // Roughness����v�Z����邽�߁A���ڂ͕s�v����
            public Vector2 uvScale = Vector2.one;
            public Vector2 uvOffset = Vector2.zero;

            public MaterialProperties(Material mat)
            {
                material = mat; // Material�Q�Ƃ�ێ�
                BaseColor = mat.HasProperty("_BaseColor") ? mat.GetColor("_BaseColor") :
                            (mat.HasProperty("_Color") ? mat.GetColor("_Color") : Color.white);
                Metallic = mat.HasProperty("_Metallic") ? mat.GetFloat("_Metallic") : 0f;
                Smoothness = mat.HasProperty("_Glossiness") ? mat.GetFloat("_Glossiness") :
                             (mat.HasProperty("_Smoothness") ? mat.GetFloat("_Smoothness") : 0.5f);
                Roughness = 1f - Smoothness;
                BaseColorMap = mat.HasProperty("_MainTex") ? mat.GetTexture("_MainTex") as Texture2D : null;
                MetallicGlossMap = mat.HasProperty("_MetallicGlossMap") ? mat.GetTexture("_MetallicGlossMap") as Texture2D : null;
                NormalMap = mat.HasProperty("_BumpMap") ? mat.GetTexture("_BumpMap") as Texture2D : null;
                MainTexST = mat.HasProperty("_MainTex_ST") ? mat.GetVector("_MainTex_ST") : new Vector4(1, 1, 0, 0);
                uvScale = new Vector2(MainTexST.x, MainTexST.y);
                uvOffset = new Vector2(MainTexST.z, MainTexST.w);
            }

            public Color GetSampledBaseColor(Vector2 uv)
            {
                Vector2 tiledUv = new Vector2(uv.x * uvScale.x + uvOffset.x, uv.y * uvScale.y + uvOffset.y);
                Color texColor = Color.white;
                if (BaseColorMap != null)
                {
                    if (BaseColorMap.isReadable) texColor = BaseColorMap.GetPixelBilinear(Mathf.Repeat(tiledUv.x, 1f), Mathf.Repeat(tiledUv.y, 1f));
                    else Debug.LogWarning($"NeuraBake: Texture '{BaseColorMap.name}' on material '{material.name}' is not readable.");
                }
                return BaseColor * texColor;
            }

            public (float sampledMetallic, float sampledRoughness) GetSampledMetallicRoughness(Vector2 uv)
            {
                Vector2 tiledUv = new Vector2(uv.x * uvScale.x + uvOffset.x, uv.y * uvScale.y + uvOffset.y);
                float finalMetallic = Metallic;
                float currentSmoothness = Smoothness;

                if (MetallicGlossMap != null)
                {
                    if (MetallicGlossMap.isReadable)
                    {
                        Color metallicGlossSample = MetallicGlossMap.GetPixelBilinear(Mathf.Repeat(tiledUv.x, 1f), Mathf.Repeat(tiledUv.y, 1f));
                        if (IsMaskMap(material, MetallicGlossMap)) // IsMaskMap��BakingCore��static���\�b�h
                        {
                            // MaskMap �̃`�����l�����蓖�Ă̓V�F�[�_�[/�p�C�v���C���Ɉˑ����邽�߁A�����ł͈��
                            // URP Lit: Metallic (G), Smoothness (A)
                            // HDRP Lit: Metallic (R), Smoothness (A)
                            // ��茘�S�ɂ���ɂ̓V�F�[�_�[�����`�F�b�N���邩�A�ݒ�őI���ł���悤�ɂ���
                            if (material.shader.name.Contains("HDRP"))
                            {
                                finalMetallic = metallicGlossSample.r;
                                currentSmoothness = metallicGlossSample.a;
                            }
                            else if (material.shader.name.Contains("URP"))
                            { // URP/Lit ������
                                finalMetallic = metallicGlossSample.g;
                                currentSmoothness = metallicGlossSample.a;
                            }
                            else
                            { // Standard Shader (Metallic setup)
                                finalMetallic *= metallicGlossSample.r;
                                currentSmoothness *= metallicGlossSample.a;
                            }
                        }
                        else // Standard Shader (Metallic setup)
                        {
                            finalMetallic *= metallicGlossSample.r;
                            currentSmoothness *= metallicGlossSample.a;
                        }
                    }
                    else Debug.LogWarning($"NeuraBake: Texture '{MetallicGlossMap.name}' on material '{material.name}' is not readable.");
                }
                return (Mathf.Clamp01(finalMetallic), 1f - Mathf.Clamp01(currentSmoothness));
            }
        }

        private class MeshDataCache
        {
            public readonly Vector3[] Vertices;
            public readonly Vector3[] Normals;
            public readonly Vector2[] UVs;
            public readonly int[] Triangles;
            public MeshDataCache(Mesh mesh)
            {
                Vertices = mesh.vertices; Normals = mesh.normals;
                UVs = (mesh.uv2 != null && mesh.uv2.Length == mesh.vertexCount) ? mesh.uv2 : mesh.uv;
                Triangles = mesh.triangles;
            }
        }

        public BakingCore(NeuraBakeSettings bakeSettings)
        {
            settings = bakeSettings ?? throw new ArgumentNullException(nameof(bakeSettings));
            sceneLights = GameObject.FindObjectsOfType<Light>().Where(l => l.isActiveAndEnabled && l.intensity > 0 && l.gameObject.activeInHierarchy).ToArray();
            emissiveSurfaces = GameObject.FindObjectsOfType<NeuraBakeEmissiveSurface>()
                                         .Where(es => es.enabled && es.bakeEmissive && es.intensity > 0 && es.gameObject.activeInHierarchy)
                                         .ToArray();
        }

        private MaterialProperties GetMaterialProperties(Material material)
        {
            lock (cacheLock) { if (!materialCache.TryGetValue(material, out MaterialProperties props)) { props = new MaterialProperties(material); materialCache[material] = props; } return props; }
        }
        private MeshDataCache GetMeshData(Mesh mesh)
        {
            lock (cacheLock) { if (!meshDataCache.TryGetValue(mesh, out MeshDataCache data)) { data = new MeshDataCache(mesh); meshDataCache[mesh] = data; } return data; }
        }

        // ILightmapRenderer�C���^�[�t�F�[�X�̎���
        public async Task<Texture2D> RenderAsync(CancellationToken token, IProgress<(float percentage, string message)> progress)
        {
            return await BakeLightmapAsync(token, progress);
        }

        public async Task<Texture2D> BakeLightmapAsync(CancellationToken token, IProgress<(float percentage, string message)> progressReporter)
        {
            MeshRenderer[] renderers = GameObject.FindObjectsOfType<MeshRenderer>()
                                               .Where(r => r.enabled && r.gameObject.isStatic && r.gameObject.activeInHierarchy).ToArray();
            if (renderers.Length == 0) { progressReporter?.Report((1f, "�x�C�N�Ώۂ̐ÓIMeshRenderer�Ȃ�")); return null; }

            MeshRenderer targetRenderer = renderers[0]; // TODO: �S�����_���[�Ή�
            Material targetMaterial = targetRenderer.sharedMaterial;
            MeshFilter meshFilter = targetRenderer.GetComponent<MeshFilter>();
            if (meshFilter?.sharedMesh == null || targetMaterial == null) { progressReporter?.Report((1f, "�Ώۃ��b�V��/�}�e���A���Ȃ�")); return null; }

            Mesh mesh = meshFilter.sharedMesh;
            MeshDataCache meshData = GetMeshData(mesh);
            if (meshData.UVs == null || meshData.UVs.Length == 0) { progressReporter?.Report((1f, "�Ώۃ��b�V����UV�Ȃ�")); return null; }

            int textureWidth = settings.atlasSize;
            int textureHeight = settings.atlasSize;
            progressReporter?.Report((0.01f, "���C�g�}�b�v�����J�n..."));

            // �����p�X�̑I�� (RTCK.NeuraBakeWindow.cs����BakingCore�𒼐ڌĂԏꍇ�͂��̕���͕s�v)
            // ���݂�BakingCore���f�t�H���gCPU�����_���[�Ƃ��ċ@�\���邽�߁A���̕���͊T�O�I�Ȃ���
            bool attemptGPU = false; // settings.rendererType == LightmapRendererType.GPU_Renderer; (Window���ŕ���ς�)
            bool attemptJobs = !attemptGPU && SystemInfo.processorCount > 1; // (Window���ŕ���ς�)

            Texture2D lightmapTexture = null;
            // �S�Ẵs�N�Z���f�[�^���i�[����z�� (�e�p�X�ł���ɏ������ނ��A�p�X�ŗL�̔z����g��)
            Color[] finalPixels = new Color[textureWidth * textureHeight];


            try
            {
                if (attemptGPU) // GPU�p�X (�X�^�u)
                {
                    progressReporter?.Report((0.05f, "GPU�������s��..."));
                    lightmapTexture = ProcessWithComputeShader(targetRenderer, mesh, targetMaterial, textureWidth, textureHeight, token, progressReporter);
                    if (lightmapTexture == null) { progressReporter?.Report((0.1f, "GPU�������s�B�t�H�[���o�b�N...")); attemptGPU = false; }
                    else { progressReporter?.Report((1f, "GPU��������")); }
                }

                if (!attemptGPU && attemptJobs) // Job System �p�X (�X�^�u)
                {
                    progressReporter?.Report((0.15f, "Job System������..."));
                    Vector3[] worldPositions = new Vector3[textureWidth * textureHeight]; // Job�ɓn�����߂̎��O�v�Z�f�[�^
                    Vector3[] worldNormals = new Vector3[textureWidth * textureHeight];   // Job�ɓn�����߂̎��O�v�Z�f�[�^

                    // worldPositions �� worldNormals ���v�Z���郍�W�b�N (�ȑO�̃R�[�h������p)
                    int validPixelCountForJob = 0;
                    Matrix4x4 localToWorld = targetRenderer.transform.localToWorldMatrix; // ���C���X���b�h�Ŏ擾

                    await Task.Run(() => Parallel.For(0, textureHeight, y_loop =>
                    {
                        if (token.IsCancellationRequested) return;
                        for (int x_loop = 0; x_loop < textureWidth; x_loop++)
                        {
                            int idx = y_loop * textureWidth + x_loop;
                            Vector2 uv = new Vector2((x_loop + 0.5f) / textureWidth, (y_loop + 0.5f) / textureHeight);
                            if (FindTriangleAndBarycentricCoords(meshData, uv, out int triIdx, out Vector3 bary))
                            {
                                int v0 = meshData.Triangles[triIdx * 3]; int v1 = meshData.Triangles[triIdx * 3 + 1]; int v2 = meshData.Triangles[triIdx * 3 + 2];
                                Vector3 lp = InterpolateVector3(meshData.Vertices[v0], meshData.Vertices[v1], meshData.Vertices[v2], bary);
                                Vector3 ln = InterpolateVector3(meshData.Normals[v0], meshData.Normals[v1], meshData.Normals[v2], bary).normalized;
                                worldPositions[idx] = localToWorld.MultiplyPoint3x4(lp);
                                worldNormals[idx] = localToWorld.MultiplyVector(ln).normalized;
                                Interlocked.Increment(ref validPixelCountForJob);
                            }
                            else
                            {
                                finalPixels[idx] = Color.clear; // Job�ŏ������Ȃ��s�N�Z���̓N���A
                            }
                        }
                    }), token);
                    if (token.IsCancellationRequested) throw new OperationCanceledException();
                    progressReporter?.Report((0.3f, $"Job�p�f�[�^�������� ({validPixelCountForJob}�s�N�Z��)"));

                    // ProcessPixelsWithJobSystem�� outputPixelData (finalPixels) �Ɍ��ʂ��������ނ悤�ɕύX
                    Texture2D jobBuiltTexture = ProcessPixelsWithJobSystem(finalPixels, textureWidth, textureHeight, worldPositions, worldNormals, token);
                    if (jobBuiltTexture != null)
                    { // jobBuiltTexture�͎��ۂɂ�finalPixels���琶�����ꂽ����
                        lightmapTexture = jobBuiltTexture; // ���̃e�N�X�`�����ŏI���ʂƂ���
                        progressReporter?.Report((1f, "Job System��������"));
                    }
                    else
                    {
                        progressReporter?.Report((0.4f, "Job System�������s�B�ʏ�CPU��..."));
                        attemptJobs = false;
                    }
                }

                if (!attemptGPU && !attemptJobs) // �ʏ�CPU���񏈗��p�X
                {
                    progressReporter?.Report((0.5f, "CPU���񏈗���..."));
                    int processedPixelCount = 0;
                    int totalValidPixelsToProcess = textureWidth * textureHeight; // �L���s�N�Z�����Ői���v�Z����
                    int reportInterval = Math.Max(1, totalValidPixelsToProcess / 100);
                    var parallelOptions = new ParallelOptions { CancellationToken = token, MaxDegreeOfParallelism = Math.Max(1, Environment.ProcessorCount - 1) };
                    mainRaycastCache.Clear();

                    await Task.Run(() =>
                    {
                        int chunkSize = 32; int numChunksX = Mathf.CeilToInt((float)textureWidth / chunkSize); int numChunksY = Mathf.CeilToInt((float)textureHeight / chunkSize);
                        Parallel.For(0, numChunksX * numChunksY, parallelOptions, chunkIdx =>
                        {
                            int cX = chunkIdx % numChunksX; int cY = chunkIdx / numChunksX;
                            int sX = cX * chunkSize; int sY = cY * chunkSize;
                            int eX = Math.Min(sX + chunkSize, textureWidth); int eY = Math.Min(sY + chunkSize, textureHeight);
                            for (int curY = sY; curY < eY; curY++)
                            {
                                for (int curX = sX; curX < eX; curX++)
                                {
                                    if (token.IsCancellationRequested) return;
                                    int pxIdx = curY * textureWidth + curX;
                                    Vector2 tcUv = new Vector2((curX + 0.5f) / textureWidth, (curY + 0.5f) / textureHeight);
                                    if (FindTriangleAndBarycentricCoords(meshData, tcUv, out int triIdx, out Vector3 bary))
                                    {
                                        int v0 = meshData.Triangles[triIdx * 3]; int v1 = meshData.Triangles[triIdx * 3 + 1]; int v2 = meshData.Triangles[triIdx * 3 + 2];
                                        Vector3 lp = InterpolateVector3(meshData.Vertices[v0], meshData.Vertices[v1], meshData.Vertices[v2], bary);
                                        Vector3 ln = InterpolateVector3(meshData.Normals[v0], meshData.Normals[v1], meshData.Normals[v2], bary).normalized;
                                        Vector2 lmUV = InterpolateVector2(meshData.UVs[v0], meshData.UVs[v1], meshData.UVs[v2], bary);
                                        Vector3 wp = targetRenderer.transform.TransformPoint(lp);
                                        Vector3 wn = targetRenderer.transform.TransformDirection(ln).normalized;
                                        Color accCol = Color.black; float accBNY = 0f; int validBNSamples = 0;
                                        for (int s = 0; s < settings.sampleCount; ++s)
                                        {
                                            Vector2 matUV = tcUv;
                                            if (settings.sampleCount > 1) { matUV = new Vector2(tcUv.x + (Random.value - 0.5f) / textureWidth, tcUv.y + (Random.value - 0.5f) / textureHeight); }
                                            var (pxCol, bnY, unocRay) = CalculatePixelColor(wp, wn, targetMaterial, lmUV, matUV);
                                            accCol += pxCol;
                                            if (settings.directional && unocRay > 0) { accBNY += bnY * unocRay; validBNSamples += unocRay; }
                                        }
                                        Color fpxCol = accCol / settings.sampleCount;
                                        if (settings.directional) { if (validBNSamples > 0) fpxCol.a = Mathf.Clamp01(0.5f + (accBNY / validBNSamples) * 0.5f); else if (settings.useAmbientOcclusion && settings.aoSampleCount > 0) fpxCol.a = 0f; else fpxCol.a = 0.5f; }
                                        else { fpxCol.a = 1f; }
                                        finalPixels[pxIdx] = fpxCol;
                                    }
                                    else { finalPixels[pxIdx] = Color.clear; }
                                    int curProc = Interlocked.Increment(ref processedPixelCount);
                                    if (curProc % reportInterval == 0 || curProc == totalValidPixelsToProcess)
                                    {
                                        float prog = 0.5f + (0.5f * curProc / totalValidPixelsToProcess);
                                        progressReporter?.Report((prog, $"CPU �s�N�Z��������: {curProc}/{totalValidPixelsToProcess}"));
                                    }
                                }
                            }
                        });
                    }, token);
                    if (token.IsCancellationRequested) throw new OperationCanceledException();

                    DilationEdgePadding(finalPixels, textureWidth, textureHeight, 8); // �G�b�W�p�f�B���O

                    lightmapTexture = new Texture2D(textureWidth, textureHeight, TextureFormat.RGBAHalf, false, true);
                    lightmapTexture.name = $"{targetRenderer.gameObject.name}_Lightmap_Baked_CPU";
                    lightmapTexture.wrapMode = TextureWrapMode.Clamp;
                    lightmapTexture.filterMode = FilterMode.Bilinear;
                    lightmapTexture.SetPixels(finalPixels);
                    lightmapTexture.Apply(true, false); //�~�b�v�}�b�v�͕s�v�Ȃ̂�false, �ǂݎ��s�ɂ͂��Ȃ�
                    progressReporter?.Report((1f, "CPU���񏈗�����"));
                }

                if (lightmapTexture == null && !token.IsCancellationRequested)
                {
                    progressReporter?.Report((1f, "�S�p�X�Ń��C�g�}�b�v�������s")); return null;
                }

                // �f�m�C�U�[�K�p
                if (settings.useDenoiser)
                {
                    progressReporter?.Report((0.9f, "�f�m�C�U�[�K�p��..."));
                    lightmapTexture = ApplyMLDenoiser(lightmapTexture);
                    progressReporter?.Report((1f, "�f�m�C�U�[�K�p����"));
                }

                return lightmapTexture;
            }
            catch (OperationCanceledException) { progressReporter?.Report((0f, "�����L�����Z��")); Debug.Log("BakingCore: �����L�����Z��"); return null; }
            catch (Exception ex) { progressReporter?.Report((0f, $"�G���[: {ex.GetType().Name}")); Debug.LogException(ex); return null; }
        }

        // ���̃��\�b�h (FindTriangleAndBarycentricCoords, CalculateBarycentricCoords, InterpolateVector3/2, CalculatePixelColor, etc.) �͕ύX�Ȃ�
        // ... (�����̃��\�b�h�̃R�[�h�͑O��̂��̂Ɠ����Ȃ̂ŏȗ�) ...
        private bool FindTriangleAndBarycentricCoords(MeshDataCache meshData, Vector2 uv, out int triangleStartVertexIndex, out Vector3 barycentricCoords)
        {
            triangleStartVertexIndex = 0; barycentricCoords = Vector3.zero;
            for (int i = 0; i < meshData.Triangles.Length / 3; i++)
            {
                Vector2 uv0 = meshData.UVs[meshData.Triangles[i * 3 + 0]]; Vector2 uv1 = meshData.UVs[meshData.Triangles[i * 3 + 1]]; Vector2 uv2 = meshData.UVs[meshData.Triangles[i * 3 + 2]];
                barycentricCoords = CalculateBarycentricCoords(uv, uv0, uv1, uv2);
                if (barycentricCoords.x >= -1e-4f && barycentricCoords.y >= -1e-4f && barycentricCoords.z >= -1e-4f && barycentricCoords.x <= 1.0001f && barycentricCoords.y <= 1.0001f && barycentricCoords.z <= 1.0001f)
                {
                    triangleStartVertexIndex = i; return true;
                }
            }
            return false;
        }
        private Vector3 CalculateBarycentricCoords(Vector2 p, Vector2 a, Vector2 b, Vector2 c)
        {
            Vector2 v0 = b - a, v1 = c - a, v2 = p - a; float d00 = Vector2.Dot(v0, v0); float d01 = Vector2.Dot(v0, v1); float d11 = Vector2.Dot(v1, v1); float d20 = Vector2.Dot(v2, v0); float d21 = Vector2.Dot(v2, v1);
            float denom = d00 * d11 - d01 * d01; if (Mathf.Abs(denom) < 1e-6f) return new Vector3(-1, -1, -1);
            float v_ = (d11 * d20 - d01 * d21) / denom; float w_ = (d00 * d21 - d01 * d20) / denom; return new Vector3(1.0f - v_ - w_, v_, w_);
        }
        private Vector3 InterpolateVector3(Vector3 v0, Vector3 v1, Vector3 v2, Vector3 barycentric) { return v0 * barycentric.x + v1 * barycentric.y + v2 * barycentric.z; }
        private Vector2 InterpolateVector2(Vector2 v0, Vector2 v1, Vector2 v2, Vector3 barycentric) { return v0 * barycentric.x + v1 * barycentric.y + v2 * barycentric.z; }

        private (Color color, float bentNormalY, int unoccludedRays) CalculatePixelColor(
            Vector3 worldPos, Vector3 worldNormal, Material material,
            Vector2 lightmapUV, Vector2 materialSampleUV)
        {
            MaterialProperties matProps = GetMaterialProperties(material);
            Color albedo = matProps.GetSampledBaseColor(materialSampleUV);
            var (metallic, roughness) = matProps.GetSampledMetallicRoughness(materialSampleUV);

            Color finalColor = Color.black;

            Color skyContribution = CalculateSkyLight(worldPos, worldNormal, albedo);
            finalColor += skyContribution;

            foreach (Light light in sceneLights)
            {
                Vector3 lightDir;
                Color lightColorAtPoint = light.color * light.intensity;
                float attenuation = 1.0f;

                if (light.type == LightType.Directional) { lightDir = -light.transform.forward; }
                else
                {
                    Vector3 lightToPointVec = worldPos - light.transform.position;
                    float distSqr = lightToPointVec.sqrMagnitude;
                    if (distSqr > light.range * light.range) continue;
                    float dist = Mathf.Sqrt(distSqr);
                    lightDir = -lightToPointVec.normalized;
                    attenuation = 1.0f / (1.0f + 25.0f * distSqr / (light.range * light.range));
                    if (light.type == LightType.Spot)
                    {
                        float spotAngle = Vector3.Angle(light.transform.forward, -lightDir);
                        if (spotAngle > light.spotAngle / 2f) continue;
                        float outerConeRad = (light.spotAngle / 2f) * Mathf.Deg2Rad;
                        float cosOuter = Mathf.Cos(outerConeRad);
                        float cosAngle = Vector3.Dot(light.transform.forward, -lightDir);
                        float spotFalloff = Mathf.Pow(Mathf.Clamp01((cosAngle - cosOuter) / (1f - cosOuter + 1e-4f)), 2f);
                        attenuation *= spotFalloff;
                    }
                }

                float maxShadowDist = (light.type == LightType.Directional) ? 2000f : Vector3.Distance(worldPos, light.transform.position);
                float shadowFactor = CalculateShadowFactor(worldPos, worldNormal, lightDir, maxShadowDist, settings.shadowSamples);

                if (shadowFactor > 0)
                {
                    Vector3 viewDir = worldNormal;
                    Color directLight = CalculateAdvancedPBRDirectLight(lightDir, viewDir, worldNormal, lightColorAtPoint * attenuation * shadowFactor, albedo, metallic, roughness);
                    finalColor += directLight;
                }
            }

            if (emissiveSurfaces.Length > 0)
            {
                finalColor += CalculateEmissiveContribution(worldPos, worldNormal, albedo);
            }

            float occlusionFactor = 0f;
            float bentNormalYComponent = worldNormal.y;
            int unoccludedRayCount = 0;

            if (settings.useAmbientOcclusion && settings.aoSampleCount > 0)
            {
                (occlusionFactor, bentNormalYComponent, unoccludedRayCount) = CalculateAmbientOcclusionAndBentNormal(worldPos, worldNormal, settings.aoSampleCount);
                finalColor *= (1f - occlusionFactor);
            }
            else if (settings.directional)
            {
                unoccludedRayCount = 1; // For bent normal calculation, even if AO is off
            }

            if (IsUVSeamEdge(GetMeshData(material.mainTexture as Mesh), lightmapUV))
            {
                // �V�[���߂��ł͓��ʂȏ������s��
                // ��: ���͂̃s�N�Z������T�d�ɐF���Ԃ���
            }

            // �F��ԕ␳�́A�ŏI�I�Ƀe�N�X�`���ɏ������ޒ��O���A�\�����ɍs���̂���ʓI�B
            // CalculatePixelColor���Ŗ���s���ƁA���Z���ς̍ۂɃ��j�A���e�B��������\��������B
            // finalColor = ApplyColorSpaceCorrection(finalColor);
            return (finalColor, bentNormalYComponent, unoccludedRayCount);
        }
        private Color CalculateEmissiveContribution(Vector3 worldPos, Vector3 worldNormal, Color albedo)
        {
            Color totalEmissiveContribution = Color.black;
            foreach (NeuraBakeEmissiveSurface emissiveSurface in emissiveSurfaces)
            {
                MeshFilter emissiveMeshFilter = emissiveSurface.GetComponent<MeshFilter>();
                if (emissiveMeshFilter?.sharedMesh == null) continue;

                Mesh emissiveMesh = emissiveMeshFilter.sharedMesh;
                int emissiveSamplePoints = 16;
                Color accumulatedLightFromThisSurface = Color.black;

                for (int i = 0; i < emissiveSamplePoints; i++)
                {
                    Vector3 b = GetRandomBarycentricCoords();
                    int randomTriangleStartIdx = Random.Range(0, emissiveMesh.triangles.Length / 3) * 3;
                    if (randomTriangleStartIdx + 2 >= emissiveMesh.triangles.Length) continue; // �z�񋫊E�`�F�b�N

                    int vIdx0 = emissiveMesh.triangles[randomTriangleStartIdx + 0];
                    int vIdx1 = emissiveMesh.triangles[randomTriangleStartIdx + 1];
                    int vIdx2 = emissiveMesh.triangles[randomTriangleStartIdx + 2];

                    // ���_�C���f�b�N�X�̋��E�`�F�b�N
                    if (vIdx0 >= emissiveMesh.vertexCount || vIdx1 >= emissiveMesh.vertexCount || vIdx2 >= emissiveMesh.vertexCount) continue;


                    Vector3 emissivePointLocal = InterpolateVector3(emissiveMesh.vertices[vIdx0], emissiveMesh.vertices[vIdx1], emissiveMesh.vertices[vIdx2], b);
                    Vector3 emissiveNormalLocal = InterpolateVector3(emissiveMesh.normals[vIdx0], emissiveMesh.normals[vIdx1], emissiveMesh.normals[vIdx2], b).normalized;
                    Vector3 emissivePointWorld = emissiveSurface.transform.TransformPoint(emissivePointLocal);
                    Vector3 emissiveNormalWorld = emissiveSurface.transform.TransformDirection(emissiveNormalLocal).normalized;

                    Vector3 dirToEmissive = emissivePointWorld - worldPos;
                    float distToEmissiveSqr = dirToEmissive.sqrMagnitude;
                    if (distToEmissiveSqr < 1e-5f) continue;
                    float distToEmissive = Mathf.Sqrt(distToEmissiveSqr);
                    Vector3 normalizedDirToEmissive = dirToEmissive / distToEmissive;

                    float NdotL_emissive = Mathf.Max(0, Vector3.Dot(worldNormal, normalizedDirToEmissive));
                    float NdotL_source = Mathf.Max(0, Vector3.Dot(emissiveNormalWorld, -normalizedDirToEmissive));

                    if (NdotL_emissive > 0 && NdotL_source > 0)
                    {
                        Ray visibilityRay = new Ray(worldPos + worldNormal * 0.001f, normalizedDirToEmissive);
                        if (!mainRaycastCache.Raycast(visibilityRay, distToEmissive - 0.002f))
                        {
                            Color emissiveLightColor = emissiveSurface.emissiveColor * emissiveSurface.intensity * settings.emissiveBoost;
                            float formFactorApprox = (NdotL_emissive * NdotL_source) / (Mathf.PI * distToEmissiveSqr + 1e-4f);
                            Color contribution = (albedo / Mathf.PI) * emissiveLightColor * formFactorApprox;
                            accumulatedLightFromThisSurface += contribution;
                        }
                    }
                }
                totalEmissiveContribution += accumulatedLightFromThisSurface / emissiveSamplePoints;
            }
            return totalEmissiveContribution;
        }
        private Vector3 GetRandomBarycentricCoords() { float r1 = Random.value; float r2 = Random.value; float sqrtR1 = Mathf.Sqrt(r1); return new Vector3(1.0f - sqrtR1, sqrtR1 * (1.0f - r2), sqrtR1 * r2); }
        private Color CalculateSkyLight(Vector3 worldPos, Vector3 worldNormal, Color albedo)
        {
            Color skyColor;
            if (RenderSettings.ambientMode == UnityEngine.Rendering.AmbientMode.Skybox && RenderSettings.skybox != null) { skyColor = RenderSettings.ambientSkyColor; }
            else if (RenderSettings.ambientMode == UnityEngine.Rendering.AmbientMode.Trilight) { float dotUp = Vector3.Dot(worldNormal, Vector3.up); float upLerp = Mathf.Clamp01(dotUp); float downLerp = Mathf.Clamp01(-dotUp); skyColor = Color.Lerp(RenderSettings.ambientEquatorColor, RenderSettings.ambientSkyColor, upLerp); skyColor = Color.Lerp(skyColor, RenderSettings.ambientGroundColor, downLerp); }
            else if (RenderSettings.ambientMode == UnityEngine.Rendering.AmbientMode.Flat) { skyColor = RenderSettings.ambientLight; }
            else { skyColor = RenderSettings.ambientSkyColor; }
            return skyColor * albedo * settings.skyIntensity * RenderSettings.ambientIntensity;
        }
        private Color CalculateAdvancedPBRDirectLight(Vector3 L, Vector3 V, Vector3 N, Color lightColor, 
            Color albedo, float metallic, float roughness, float ior = 1.45f)
        {
            // ��{�I��PBR�v�Z�i�����R�[�h�j
            L = L.normalized; V = V.normalized; N = N.normalized;
            Vector3 H = (L + V).normalized;
            float NdotL = Mathf.Max(0f, Vector3.Dot(N, L));
            if (NdotL <= 0f) return Color.black;
            
            float NdotV = Mathf.Max(0f, Vector3.Dot(N, V));
            float alpha = roughness * roughness;
            
            // IOR���l������F0�v�Z
            float f0 = Mathf.Pow((ior - 1) / (ior + 1), 2);
            Color F0 = Color.Lerp(new Color(f0, f0, f0), albedo, metallic);
            
            // ���d�U��GGX
            float Vis = ImprovedVisibilityTerm(NdotL, NdotV, roughness);
            float D = ImprovedGGXDistribution(N, H, alpha);
            Color F = ImprovedFresnelTerm(Mathf.Max(0f, Vector3.Dot(H, V)), F0);
            
            // �G�l���M�[�ۑ����l�������g�U����
            Color kd = Color.Lerp(Color.white - F, Color.black, metallic);
            float energyFactor = EnergyCompensation(roughness, NdotV);
            Color diffuse = kd * albedo / Mathf.PI * energyFactor;
            
            // ���d�U�����l���������ʔ���
            Color specular = (D * F * Vis) / (4f * NdotL * NdotV + 1e-6f);
            
            return (diffuse + specular) * lightColor * NdotL;
        }

        private float EnergyCompensation(float roughness, float NdotV)
        {
            // Disney�g�U���f���Ɋ�Â��G�l���M�[�⏞
            float energyBias = Mathf.Lerp(0, 0.5f, roughness);
            float energyFactor = Mathf.Lerp(1.0f, 1.0f / 1.51f, roughness);
            float fd90 = energyBias + 2.0f * roughness * NdotV * NdotV;
            float f0 = 1.0f;
            float lightScatter = f0 + (fd90 - f0) * Mathf.Pow(1.0f - NdotV, 5.0f);
            return lightScatter * energyFactor;
        }
        private Color Fresnel_Schlick(float cosTheta, Color F0) { return F0 + (Color.white - F0) * Mathf.Pow(1f - Mathf.Max(0, cosTheta), 5f); } // Ensure cosTheta is not negative
        private float GGX_Distribution(Vector3 N, Vector3 H, float alpha) { float NdotH = Mathf.Max(0f, Vector3.Dot(N, H)); float alphaSq = alpha * alpha; float denom = NdotH * NdotH * (alphaSq - 1f) + 1f; return alphaSq / (Mathf.PI * denom * denom + 1e-7f); } // Added epsilon to denominator
        private float Smith_Visibility_JointGGX(Vector3 N, Vector3 V, Vector3 L, float alpha) { float NdotV_ = Mathf.Max(Vector3.Dot(N, V), 0.0f) + 1e-5f; float NdotL_ = Mathf.Max(Vector3.Dot(N, L), 0.0f) + 1e-5f; float roughness_sq = alpha * alpha; float G_SmithV = (2.0f * NdotV_) / (NdotV_ + Mathf.Sqrt(roughness_sq + (1.0f - roughness_sq) * NdotV_ * NdotV_)); float G_SmithL = (2.0f * NdotL_) / (NdotL_ + Mathf.Sqrt(roughness_sq + (1.0f - roughness_sq) * NdotL_ * NdotL_)); return G_SmithV * G_SmithL; }


        private (float occlusion, float bentNormalY, int unoccludedRays) CalculateAmbientOcclusionAndBentNormal(Vector3 position, Vector3 normal, int sampleCount)
        {
            if (sampleCount <= 0) return (0f, normal.y, 0);

            float sumOcclusionFactor = 0f; // �C��: float�^�Ȃ̂�Interlocked�͎g���Ȃ��B���lock����B
            Vector3 accumulatedUnoccludedNormal = Vector3.zero;
            int unoccludedRayCountAtomic = 0; // �������Interlocked�\
            float maxAoDistance = 2.0f;

            Vector3[] sampleDirections = new Vector3[sampleCount];
            for (int i = 0; i < sampleCount; i++) { Vector3 randomDir = Random.onUnitSphere; if (Vector3.Dot(randomDir, normal) < 0) randomDir = -randomDir; sampleDirections[i] = randomDir; }

            Parallel.For(0, sampleCount, i =>
            {
                Ray aoRay = new Ray(position + normal * 0.001f, sampleDirections[i]);
                if (!mainRaycastCache.Raycast(aoRay, maxAoDistance))
                {
                    lock (cacheLock) { accumulatedUnoccludedNormal += sampleDirections[i]; }
                    Interlocked.Increment(ref unoccludedRayCountAtomic);
                }
                else
                {
                    lock (cacheLock) { sumOcclusionFactor += 1f; } // ���C��: lock���g����sumOcclusionFactor���X�V
                }
            });

            float finalOcclusion = (sampleCount > 0) ? sumOcclusionFactor / sampleCount : 0f;
            float finalBentNormalY = normal.y;
            if (unoccludedRayCountAtomic > 0)
            {
                // accumulatedUnoccludedNormal �� lock ���ōX�V���ꂽ�̂ŁA�ǂݎ��� lock ���邩�A
                // ���̎��_�ł� Parallel.For ���������Ă���̂ŁA���C���X���b�h����̓ǂݎ��͈��S�B
                // �������A�������݂ƃ^�C�~���O���V�r�A�ȏꍇ�� lock �𐄏��B
                // �����ł� Parallel.For ������̏����Ȃ̂� lock �Ȃ��œǂݎ��B
                finalBentNormalY = (accumulatedUnoccludedNormal / unoccludedRayCountAtomic).normalized.y;
            }
            return (finalOcclusion, finalBentNormalY, unoccludedRayCountAtomic);
        }
        private static bool IsMaskMap(Material material, Texture2D texture) { if (texture == null || material?.shader == null) return false; bool isHDRP = material.shader.name.Contains("HDRP") || material.shader.name.Contains("High Definition"); bool isURP = material.shader.name.Contains("URP") || material.shader.name.Contains("Universal"); bool hasMaskMapProperty = material.HasProperty("_MaskMap"); return isHDRP || hasMaskMapProperty || (texture.name.Contains("_MaskMap") && isURP); }
        private Color ApplyColorSpaceCorrection(Color color) { if (QualitySettings.activeColorSpace == ColorSpace.Gamma) return new Color(Mathf.LinearToGammaSpace(color.r), Mathf.LinearToGammaSpace(color.g), Mathf.LinearToGammaSpace(color.b), color.a); return color; }
        private float CalculateShadowFactor(Vector3 worldPos, Vector3 worldNormal, Vector3 lightDir, float maxShadowDist, int shadowSamples)
        {
            if (shadowSamples <= 0) return 1.0f; float shadowAccumulator = 0f;
            if (shadowSamples == 1) { Ray sr = new Ray(worldPos + worldNormal * 0.001f, -lightDir); return mainRaycastCache.Raycast(sr, maxShadowDist - 0.002f) ? 0.0f : 1.0f; }
            float[] sampleResults = new float[shadowSamples];
            Parallel.For(0, shadowSamples, i =>
            {
                Vector3 jld = lightDir; if (shadowSamples > 1) { jld = (lightDir + (Random.onUnitSphere * 0.05f)).normalized; } // 0.05f�͒����l
                Ray sr = new Ray(worldPos + worldNormal * 0.001f, -jld); sampleResults[i] = mainRaycastCache.Raycast(sr, maxShadowDist - 0.002f) ? 0.0f : 1.0f;
            });
            for (int i = 0; i < shadowSamples; i++) shadowAccumulator += sampleResults[i];
            return shadowAccumulator / shadowSamples;
        }

        // Job System�p Job��`
        [BurstCompile]
        private struct ProcessPixelChunkJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float3> WorldPositions;
            [ReadOnly] public NativeArray<float3> WorldNormals;
            [ReadOnly] public float4 SkyColorParam;
            [ReadOnly] public float SkyIntensityParam;
            [ReadOnly] public float AmbientIntensityParam;
            public NativeArray<float4> ResultColors;

            public void Execute(int index)
            {
                if (index >= WorldPositions.Length) return;
                float3 worldPos = WorldPositions[index]; float3 worldNormal = WorldNormals[index];
                if (math.lengthsq(worldNormal) < 0.0001f) worldNormal = new float3(0, 1, 0); else worldNormal = math.normalize(worldNormal);
                float dotUp = math.dot(worldNormal, new float3(0, 1, 0)); float upLerp = math.clamp(dotUp, 0f, 1f);
                float4 skyContribution = SkyColorParam * upLerp * SkyIntensityParam * AmbientIntensityParam;
                // TODO: Implement full lighting within the job
                ResultColors[index] = skyContribution;
            }
        }
        private Texture2D ProcessPixelsWithJobSystem(Color[] outputPixelData, int width, int height, Vector3[] worldPositions, Vector3[] worldNormals, CancellationToken token)
        {
            int totalPixels = width * height;
            if (worldPositions.Length != totalPixels || worldNormals.Length != totalPixels) { Debug.LogError("Job System: Array length mismatch."); return null; }

            var positionsNative = new NativeArray<float3>(totalPixels, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
            var normalsNative = new NativeArray<float3>(totalPixels, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
            var colorsNative = new NativeArray<float4>(totalPixels, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
            Texture2D resultTexture = null;
            try
            {
                for (int i = 0; i < totalPixels; i++) { if (token.IsCancellationRequested) throw new OperationCanceledException(token); positionsNative[i] = worldPositions[i]; normalsNative[i] = worldNormals[i]; }
                var job = new ProcessPixelChunkJob
                {
                    WorldPositions = positionsNative,
                    WorldNormals = normalsNative,
                    SkyColorParam = new float4(RenderSettings.ambientSkyColor.r, RenderSettings.ambientSkyColor.g, RenderSettings.ambientSkyColor.b, RenderSettings.ambientSkyColor.a),
                    SkyIntensityParam = settings.skyIntensity,
                    AmbientIntensityParam = RenderSettings.ambientIntensity,
                    ResultColors = colorsNative
                };
                JobHandle handle = job.Schedule(totalPixels, 64); handle.Complete();
                if (token.IsCancellationRequested) throw new OperationCanceledException(token);
                resultTexture = new Texture2D(width, height, TextureFormat.RGBAHalf, false, true);
                Color[] tempColors = new Color[totalPixels];
                for (int i = 0; i < totalPixels; i++) { if (token.IsCancellationRequested) throw new OperationCanceledException(token); float4 c = colorsNative[i]; tempColors[i] = new Color(c.x, c.y, c.z, c.w); }
                resultTexture.SetPixels(tempColors); resultTexture.Apply(true, false); // Apply non-readable to save memory after CPU access
                Array.Copy(tempColors, outputPixelData, totalPixels); // outputPixelData �ɂ����ʂ��R�s�[
                return resultTexture;
            }
            catch (Exception ex) { Debug.LogError($"Job System error: {ex.Message}"); return null; }
            finally { if (positionsNative.IsCreated) positionsNative.Dispose(); if (normalsNative.IsCreated) normalsNative.Dispose(); if (colorsNative.IsCreated) colorsNative.Dispose(); }
        }

        // GPU���� (�R���s���[�g�V�F�[�_�[) �X�^�u
        private Texture2D ProcessWithComputeShader(MeshRenderer targetRenderer, Mesh mesh, Material targetMaterial, int textureWidth, int textureHeight, CancellationToken token, IProgress<(float percentage, string message)> progressReporter)
        {
            progressReporter?.Report((0.06f, "�R���s���[�g�V�F�[�_�[����..."));
            ComputeShader computeShader = InitializeComputeShader();
            if (computeShader == null) { Debug.LogWarning("CS 'NeuraBakeGPURenderer' not found in Resources."); return null; }

            RenderTexture outputRT = null;
            List<ComputeBuffer> buffersToRelease = new List<ComputeBuffer>();
            try
            {
                MeshDataCache meshData = GetMeshData(mesh);
                outputRT = new RenderTexture(textureWidth, textureHeight, 0, RenderTextureFormat.ARGBHalf) { enableRandomWrite = true }; outputRT.Create();

                Action<object[]> addBuffer = (dataArr) =>
                {
                    if (dataArr[0] is Vector3[]) { var buf = new ComputeBuffer(((Vector3[])dataArr[0]).Length, sizeof(float) * 3); buf.SetData((Vector3[])dataArr[0]); buffersToRelease.Add(buf); computeShader.SetBuffer((int)dataArr[2], (string)dataArr[1], buf); }
                    else if (dataArr[0] is Vector2[]) { var buf = new ComputeBuffer(((Vector2[])dataArr[0]).Length, sizeof(float) * 2); buf.SetData((Vector2[])dataArr[0]); buffersToRelease.Add(buf); computeShader.SetBuffer((int)dataArr[2], (string)dataArr[1], buf); }
                    else if (dataArr[0] is int[]) { var buf = new ComputeBuffer(((int[])dataArr[0]).Length, sizeof(int)); buf.SetData((int[])dataArr[0]); buffersToRelease.Add(buf); computeShader.SetBuffer((int)dataArr[2], (string)dataArr[1], buf); }
                    else if (dataArr[0] is Vector4[]) { var buf = new ComputeBuffer(((Vector4[])dataArr[0]).Length > 0 ? ((Vector4[])dataArr[0]).Length : 1, sizeof(float) * 4); buf.SetData(((Vector4[])dataArr[0]).Length > 0 ? (Vector4[])dataArr[0] : new Vector4[] { Vector4.zero }); buffersToRelease.Add(buf); computeShader.SetBuffer((int)dataArr[2], (string)dataArr[1], buf); }
                };

                int kernel = computeShader.FindKernel("CSMain");
                addBuffer(new object[] { meshData.Vertices, "Vertices", kernel }); addBuffer(new object[] { meshData.Normals, "Normals", kernel }); addBuffer(new object[] { meshData.UVs, "UVs", kernel }); addBuffer(new object[] { meshData.Triangles, "Triangles", kernel });

                Vector4[] lightPos = new Vector4[Math.Max(1, sceneLights.Length)]; Vector4[] lightCol = new Vector4[Math.Max(1, sceneLights.Length)]; Vector4[] lightDir = new Vector4[Math.Max(1, sceneLights.Length)]; Vector4[] lightPar = new Vector4[Math.Max(1, sceneLights.Length)];
                for (int i = 0; i < sceneLights.Length; i++) { lightPos[i] = sceneLights[i].transform.position; lightCol[i] = sceneLights[i].color * sceneLights[i].intensity; lightDir[i] = sceneLights[i].transform.forward; lightPar[i] = new Vector4((int)sceneLights[i].type, sceneLights[i].range, sceneLights[i].spotAngle, sceneLights[i].intensity); }
                if (sceneLights.Length == 0) { lightPos[0] = lightCol[0] = lightDir[0] = lightPar[0] = Vector4.zero; }

                addBuffer(new object[] { lightPos, "LightPositions", kernel }); addBuffer(new object[] { lightCol, "LightColors", kernel }); addBuffer(new object[] { lightDir, "LightDirections", kernel }); addBuffer(new object[] { lightPar, "LightParams", kernel });

                computeShader.SetTexture(kernel, "Result", outputRT);
                computeShader.SetInt("TextureWidth", textureWidth); computeShader.SetInt("TextureHeight", textureHeight);
                computeShader.SetInt("VertexCount", meshData.Vertices.Length); computeShader.SetInt("TriangleCount", meshData.Triangles.Length / 3);
                computeShader.SetMatrix("LocalToWorldMatrix", targetRenderer.transform.localToWorldMatrix);
                computeShader.SetInt("LightCountActual", sceneLights.Length); // ���ۂ̃��C�g��

                computeShader.SetVector("CS_AmbientSkyColor", RenderSettings.ambientSkyColor); computeShader.SetVector("CS_AmbientEquatorColor", RenderSettings.ambientEquatorColor); computeShader.SetVector("CS_AmbientGroundColor", RenderSettings.ambientGroundColor);
                computeShader.SetFloat("CS_AmbientIntensity", RenderSettings.ambientIntensity); computeShader.SetInt("CS_AmbientMode", (int)RenderSettings.ambientMode);
                computeShader.SetFloat("CS_SkyIntensity", settings.skyIntensity);
                computeShader.SetInt("CS_SampleCount", settings.sampleCount); computeShader.SetInt("CS_AoSampleCount", settings.aoSampleCount);
                computeShader.SetInt("CS_ShadowSamples", settings.shadowSamples); computeShader.SetBool("CS_UseAO", settings.useAmbientOcclusion); computeShader.SetBool("CS_Directional", settings.directional);

                int tgX = Mathf.CeilToInt(textureWidth / 8.0f); int tgY = Mathf.CeilToInt(textureHeight / 8.0f);
                computeShader.Dispatch(kernel, tgX, tgY, 1);

                progressReporter?.Report((0.08f, "GPU�v�Z�����A�e�N�X�`���ϊ�..."));
                Texture2D resTex = ConvertRenderTextureToTexture2D(outputRT);
                if (resTex != null) resTex.name = $"{targetRenderer.gameObject.name}_Lightmap_Baked_GPU";
                return resTex;
            }
            catch (Exception e) { Debug.LogError($"CS Error: {e.Message}\n{e.StackTrace}"); return null; }
            finally { outputRT?.Release(); foreach (var b in buffersToRelease) b?.Release(); }
        }

        private ComputeShader InitializeComputeShader()
        {
            ComputeShader shader = Resources.Load<ComputeShader>("NeuraBakeGPURenderer");
            if (shader == null)
            {
                Debug.LogError("NeuraBakeGPURenderer compute shader not found!");
                return null;
            }
            
            // ���x�ȃ����_�����O�ݒ�
            int kernel = shader.FindKernel("CSMain");
            shader.SetBool("UseImportanceSampling", true);
            shader.SetBool("UseDenoising", settings.useDenoiser);
            shader.SetInt("MaxBounces", settings.bounceCount);
            shader.SetFloat("FilterRadius", 0.01f); // ���E�����p�t�B���^���a
            
            return shader;
        }

        // ���ǂ��ꂽ�G�b�W�p�f�B���O����
        private static void DilationEdgePadding(Color[] pixels, int width, int height, int iterations = 2)
        {
            if (pixels == null || pixels.Length != width * height) return;
            Color[] tempPixels = new Color[pixels.Length]; // ��Ɨp�z��
            Color[] originalPixels = new Color[pixels.Length]; // ���̃s�N�Z����Ԃ�ۑ�
            Array.Copy(pixels, originalPixels, pixels.Length);

            // UV�������ʂ��邽�߂̃}�X�N (0=������, 1=�L���ȃs�N�Z��, 2=�p�f�B���O���ꂽ�s�N�Z��)
            byte[] islandMask = new byte[pixels.Length];
            for (int i = 0; i < pixels.Length; i++)
            {
                if (pixels[i].a > 0.001f) islandMask[i] = 1; // ���ɗL���ȃs�N�Z��
            }

            for (int iter = 0; iter < iterations; iter++)
            {
                Array.Copy(pixels, tempPixels, pixels.Length); // ���݂̃s�N�Z����Ԃ��R�s�[

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int currentIndex = y * width + x;
                        if (islandMask[currentIndex] != 0) continue; // ���ɏ����ς݂̃s�N�Z���̓X�L�b�v

                        // ����UV���ɑ�����אڃs�N�Z���݂̂���F�����W
                        Color accumulatedColor = Color.black;
                        float accumulatedWeight = 0;
                        int validNeighborCount = 0;

                        for (int dy = -1; dy <= 1; dy++)
                        {
                            for (int dx = -1; dx <= 1; dx++)
                            {
                                if (dx == 0 && dy == 0) continue; // �������g�̓X�L�b�v
                                int nx = x + dx;
                                int ny = y + dy;
                                if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                                {
                                    int neighborIndex = ny * width + nx;
                                    if (islandMask[neighborIndex] == 1) // ���X�L���������s�N�Z���̂ݎQ��
                                    {
                                        // �����Ɋ�Â��d�ݕt��
                                        float weight = (dx == 0 || dy == 0) ? 1.0f : 0.7071f; // �΂߂͋�������2�Ȃ̂ŏd�݂�������
                                        accumulatedColor += pixels[neighborIndex] * weight;
                                        accumulatedWeight += weight;
                                        validNeighborCount++;
                                    }
                                }
                            }
                        }

                        // �L���ȗאڃs�N�Z��������ꍇ�̂ݐF��K�p
                        if (validNeighborCount > 0)
                        {
                            // �d�ݕt�����ςŐF���v�Z
                            tempPixels[currentIndex] = accumulatedColor / accumulatedWeight;
                            
                            // �A���t�@�l�͌��̃s�N�Z������ێ��i��������1.0�ɐݒ�j
                            if (originalPixels[currentIndex].a < 0.001f)
                            {
                                tempPixels[currentIndex].a = 1.0f;
                            }
                            
                            islandMask[currentIndex] = 2; // ���̃s�N�Z���̓p�f�B���O���ꂽ
                        }
                    }
                }

                Array.Copy(tempPixels, pixels, pixels.Length); // �ύX�����̔z��ɔ��f
            }
            
            // �Ō�ɁA���X���݂��Ă����s�N�Z���̐F��ێ��i�p�f�B���O�ŏ㏑�����Ȃ��j
            for (int i = 0; i < pixels.Length; i++)
            {
                if (islandMask[i] == 1)
                {
                    pixels[i] = originalPixels[i];
                }
            }
        }

        // UV�����ʂ����O�ɍs�����߂̊֐�
        private static Dictionary<int, int> IdentifyUVIslands(MeshDataCache meshData)
        {
            Dictionary<int, int> triangleToIsland = new Dictionary<int, int>();
            Dictionary<Vector2, List<int>> vertexUVToTriangles = new Dictionary<Vector2, List<int>>();
            int islandCount = 0;
            
            // �e���_UV����Q�Ƃ���O�p�`���L�^
            for (int i = 0; i < meshData.Triangles.Length / 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    int vertIndex = meshData.Triangles[i * 3 + j];
                    Vector2 uv = meshData.UVs[vertIndex];
                    
                    if (!vertexUVToTriangles.TryGetValue(uv, out var triangles))
                    {
                        triangles = new List<int>();
                        vertexUVToTriangles[uv] = triangles;
                    }
                    triangles.Add(i);
                }
            }
            
            // �e�O�p�`�ɂ��ē�������
            HashSet<int> processedTriangles = new HashSet<int>();
            for (int i = 0; i < meshData.Triangles.Length / 3; i++)
            {
                if (processedTriangles.Contains(i)) continue;
                
                // �V���������J�n
                islandCount++;
                Queue<int> queue = new Queue<int>();
                queue.Enqueue(i);
                processedTriangles.Add(i);
                triangleToIsland[i] = islandCount;
                
                while (queue.Count > 0)
                {
                    int triangleIndex = queue.Dequeue();
                    // ���̎O�p�`�̊e���_����ڑ�����O�p�`������
                    for (int j = 0; j < 3; j++)
                    {
                        int vertIndex = meshData.Triangles[triangleIndex * 3 + j];
                        Vector2 uv = meshData.UVs[vertIndex];
                        
                        foreach (int connectedTriangle in vertexUVToTriangles[uv])
                        {
                            if (!processedTriangles.Contains(connectedTriangle))
                            {
                                processedTriangles.Add(connectedTriangle);
                                triangleToIsland[connectedTriangle] = islandCount;
                                queue.Enqueue(connectedTriangle);
                            }
                        }
                    }
                }
            }
            
            return triangleToIsland;
        }

        // UV�V�[�������o���ă}�X�N����w���p�[�֐�
        private static bool IsUVSeamEdge(MeshDataCache meshData, Vector2 uv, float threshold = 0.01f)
        {
            // UV�̍��W�����b�V���̕ӂɋ߂����ǂ������m�F
            for (int i = 0; i < meshData.Triangles.Length; i += 3)
            {
                Vector2 uv0 = meshData.UVs[meshData.Triangles[i]];
                Vector2 uv1 = meshData.UVs[meshData.Triangles[i + 1]];
                Vector2 uv2 = meshData.UVs[meshData.Triangles[i + 2]];
                
                // �O�p�`�̃G�b�W�ɋ߂����`�F�b�N
                float dist1 = PointToLineDistance(uv, uv0, uv1);
                float dist2 = PointToLineDistance(uv, uv1, uv2);
                float dist3 = PointToLineDistance(uv, uv2, uv0);
                
                if (dist1 < threshold || dist2 < threshold || dist3 < threshold)
                {
                    return true;
                }
            }
            return false;
        }

        private static float PointToLineDistance(Vector2 p, Vector2 a, Vector2 b)
        {
            Vector2 ab = b - a;
            Vector2 ap = p - a;
            
            if (Vector2.Dot(ab, ap) <= 0) return ap.magnitude; // �_P�͐����̊O�iA�����O�j
            
            Vector2 bp = p - b;
            if (Vector2.Dot(ab, bp) >= 0) return bp.magnitude; // �_P�͐����̊O�iB������j
            
            return Mathf.Abs(ab.x * ap.y - ab.y * ap.x) / ab.magnitude; // �_�Ɛ��̋���
        }

        // ��ԕ����ɂ��œK��
        private void BuildAccelerationStructure()
        {
            // BVH�i���E�{�����[���K�w�j�\�z
            BVHNode rootNode = new BVHNode();
            List<Triangle> allTriangles = new List<Triangle>();
            
            // ���b�V������O�p�`�f�[�^���W
            foreach (MeshRenderer renderer in staticRenderers)
            {
                Mesh mesh = renderer.GetComponent<MeshFilter>().sharedMesh;
                if (mesh == null) continue;
                
                Matrix4x4 localToWorld = renderer.transform.localToWorldMatrix;
                for (int i = 0; i < mesh.triangles.Length; i += 3)
                {
                    Triangle tri = new Triangle(
                        localToWorld.MultiplyPoint(mesh.vertices[mesh.triangles[i]]),
                        localToWorld.MultiplyPoint(mesh.vertices[mesh.triangles[i + 1]]),
                        localToWorld.MultiplyPoint(mesh.vertices[mesh.triangles[i + 2]]),
                        renderer
                    );
                    allTriangles.Add(tri);
                }
            }
            
            // BVH�̍\�z�i�����A���S���Y���j
            rootNode.Build(allTriangles, 0);
            accelerationStructure = rootNode;
        }

        // �v���O���b�V�u�����_�����O�̂��߂̐ݒ�
        private IEnumerator RenderProgressively(int targetWidth, int targetHeight)
        {
            // �i�K�I�ɉ𑜓x���グ��
            int[] resolutionSteps = { 64, 128, 256, 512, targetWidth };
            int currentSamples = 0;
            
            foreach (int resolution in resolutionSteps)
            {
                // ��𑜓x�Ń����_�����O
                int scaledWidth = Mathf.Min(resolution, targetWidth);
                int scaledHeight = Mathf.Min(resolution, targetHeight);
                
                Color[] pixels = new Color[scaledWidth * scaledHeight];
                RenderWithSamples(pixels, scaledWidth, scaledHeight, 4); // ���Ȃ��T���v�����őf����
                
                // �v���r���[�\��
                previewTexture = new Texture2D(scaledWidth, scaledHeight);
                previewTexture.SetPixels(pixels);
                previewTexture.Apply();
                
                yield return new WaitForSeconds(0.1f);
                
                // �\���ȉ𑜓x�ɒB������A�T���v�����𑝂₷�����ɐ؂�ւ�
                if (scaledWidth == targetWidth)
                {
                    while (currentSamples < settings.sampleCount)
                    {
                        int samplesToAdd = Mathf.Min(4, settings.sampleCount - currentSamples);
                        currentSamples += samplesToAdd;
                        
                        RenderWithSamples(pixels, scaledWidth, scaledHeight, samplesToAdd, accumulate: true);
                        previewTexture.SetPixels(pixels);
                        previewTexture.Apply();
                        
                        progressReporter?.Report(((float)currentSamples / settings.sampleCount, 
                            $"�v���O���b�V�u�T���v�����O: {currentSamples}/{settings.sampleCount}"));
                        yield return new WaitForSeconds(0.2f);
                    }
                }
            }
        }

        // ML-Ops�x�[�X�̃f�m�C�U�[�i�T�O�R�[�h�j
        private Texture2D ApplyMLDenoiser(Texture2D noisyTexture)
        {
            // �o�����g���b�N�o�b�t�@�̏���
            RenderTexture albedoBuffer = CreateAuxiliaryBuffer(noisyTexture.width, noisyTexture.height);
            RenderTexture normalBuffer = CreateAuxiliaryBuffer(noisyTexture.width, noisyTexture.height);
            RenderTexture positionBuffer = CreateAuxiliaryBuffer(noisyTexture.width, noisyTexture.height);
            
            // G-Buffer�f�[�^�𐶐�
            GenerateGBuffers(albedoBuffer, normalBuffer, positionBuffer);
            
            // AI/ML���f�����g�p�����f�m�C�W���O
            // ��: ���ۂ̎����ł�Barracuda/TensorFlow�Ȃǂ��g�p
            ModelExecutioner modelExecutioner = new ModelExecutioner("Assets/RTCK/NeuraBake/ML/denoiser_model.onnx");
            Dictionary<string, Texture> inputs = new Dictionary<string, Texture>()
            {
                { "noisy", noisyTexture },
                { "albedo", albedoBuffer },
                { "normal", normalBuffer },
                { "position", positionBuffer }
            };
            
            RenderTexture result = modelExecutioner.Execute(inputs);
            
            // ���ʂ�Texture2D�ɕϊ����ĕԂ�
            return ConvertRenderTextureToTexture2D(result);
        }
    }

    // IES�v���t�@�C���Ή��̌����N���X
    public class IESLight
    {
        private float[] intensityData;
        private int verticalSamples;
        private int horizontalSamples;
        
        public IESLight(string iesFilePath)
        {
            // IES�t�@�C���̉�͂Ɠǂݍ���
            ParseIESFile(iesFilePath);
        }
        
        public float GetIntensity(Vector3 direction)
        {
            // ��������p�x���v�Z
            float verticalAngle = Mathf.Acos(direction.y) * Mathf.Rad2Deg;
            float horizontalAngle = Mathf.Atan2(direction.z, direction.x) * Mathf.Rad2Deg;
            
            // �p�x����f�[�^�C���f�b�N�X���v�Z
            int vertIndex = Mathf.FloorToInt(verticalAngle / 180f * (verticalSamples - 1));
            int horizIndex = Mathf.FloorToInt((horizontalAngle + 180f) / 360f * (horizontalSamples - 1));
            
            // �C���f�b�N�X���N�����v
            vertIndex = Mathf.Clamp(vertIndex, 0, verticalSamples - 1);
            horizIndex = Mathf.Clamp(horizIndex, 0, horizontalSamples - 1);
            
            return intensityData[vertIndex * horizontalSamples + horizIndex];
        }
        
        // IES�t�@�C����̓��\�b�h
        private void ParseIESFile(string path) { /* IES�t�@�C����̓��W�b�N */ }
    }
}